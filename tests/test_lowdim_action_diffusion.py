"""Unit checks for the low-dimensional teaching diffusion model."""

import numpy as np
import pytest
import torch

from robot_learning.behavior_cloning import ObservationNormalization
from robot_learning.lowdim_action_diffusion import (
    DiffusionChunkPolicy,
    LowDimActionDiffusion,
    load_diffusion_checkpoint,
    masked_noise_mse,
    save_diffusion_checkpoint,
)


def test_cosine_schedule_corrupts_actions_and_add_noise_matches_formula() -> None:
    model = LowDimActionDiffusion(7, 2, horizon=4, diffusion_steps=16, hidden_size=16)
    assert torch.all(model.alpha_bars[1:] < model.alpha_bars[:-1])
    assert float(model.alpha_bars[-1]) < 0.01
    actions = torch.ones((2, 4, 2))
    noise = torch.zeros_like(actions)
    steps = torch.tensor([0, 15], dtype=torch.long)
    actual = model.add_noise(actions, steps, noise)
    expected = model.alpha_bars[steps].sqrt()[:, None, None].expand_as(actions)
    torch.testing.assert_close(actual, expected)


def test_masked_noise_loss_does_not_score_padding() -> None:
    target = torch.zeros((1, 3, 2))
    predicted = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]]])
    valid = torch.tensor([[True, True, False]])
    assert masked_noise_mse(predicted, target, valid).item() == 2.5
    with pytest.raises(ValueError, match="valid actions"):
        masked_noise_mse(predicted, target, torch.zeros_like(valid))


def test_sampling_is_bounded_and_reproducible_for_same_seed() -> None:
    model = LowDimActionDiffusion(7, 2, horizon=4, diffusion_steps=8, hidden_size=16)
    observations = torch.zeros((3, 7))
    first = model.sample(observations, generator=torch.Generator().manual_seed(42))
    second = model.sample(observations, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(first, second)
    assert first.shape == (3, 4, 2)
    assert torch.isfinite(first).all()
    assert torch.all(first.abs() <= 1)


def test_policy_executes_prefix_and_replans(monkeypatch: pytest.MonkeyPatch) -> None:
    model = LowDimActionDiffusion(7, 2, horizon=4, diffusion_steps=8, hidden_size=16)
    calls = []

    def fake_sample(observations: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
        del generator
        calls.append(observations.clone())
        return torch.tensor([[[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]]])

    monkeypatch.setattr(model, "sample", fake_sample)
    normalization = ObservationNormalization(np.zeros(7), np.ones(7))
    policy = DiffusionChunkPolicy(model, normalization, execution_horizon=2)
    actions = [policy(np.zeros(7))[0] for _ in range(3)]
    np.testing.assert_allclose(actions, [0.1, 0.2, 0.1])
    assert len(calls) == 2
    policy.reset(seed=5)
    np.testing.assert_allclose(policy(np.zeros(7)), [0.1, 0.0])
    assert len(calls) == 3
    with pytest.raises(ValueError, match="execution_horizon"):
        DiffusionChunkPolicy(model, normalization, execution_horizon=5)


def test_checkpoint_preserves_model_normalization_and_dataset_id(tmp_path) -> None:
    model = LowDimActionDiffusion(7, 2, horizon=4, diffusion_steps=8, hidden_size=16)
    normalization = ObservationNormalization(np.arange(7, dtype=np.float32), np.ones(7))
    path = tmp_path / "model.pt"
    save_diffusion_checkpoint(path, model, normalization,
                              source_dataset_sha256="example-hash", seed=42)
    loaded, loaded_normalization, artifact = load_diffusion_checkpoint(path)
    assert artifact["source_dataset_sha256"] == "example-hash"
    np.testing.assert_array_equal(loaded_normalization.mean, normalization.mean)
    observations = torch.zeros((2, 7))
    noisy = torch.zeros((2, 4, 2))
    steps = torch.tensor([0, 7])
    torch.testing.assert_close(
        model(observations, noisy, steps), loaded(observations, noisy, steps)
    )
