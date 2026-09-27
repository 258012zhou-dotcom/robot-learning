"""Unit tests for the ACT-style execution path and checkpoint reload."""

import numpy as np
import pytest
import torch

from robot_learning.act_lowdim import ACTLowDim, ACTLowDimOutput
from robot_learning.act_lowdim_policy import (
    ACTLowDimPolicy,
    load_act_lowdim_checkpoint,
)
from robot_learning.behavior_cloning import ObservationNormalization


class CountingACT(ACTLowDim):
    def __init__(self) -> None:
        super().__init__(1, 1, 3, hidden_size=8, latent_size=2, attention_heads=2)
        self.query_count = 0

    def forward(
        self,
        observations: torch.Tensor,
        target_actions: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> ACTLowDimOutput:
        assert target_actions is None and valid_mask is None
        self.query_count += 1
        start = self.query_count * 0.1
        actions = torch.tensor([[[start], [start + 0.3], [start + 0.6]]])
        return ACTLowDimOutput(actions, None, None)


def test_latest_first_action_queries_without_demonstration() -> None:
    model = CountingACT()
    norm = ObservationNormalization(np.zeros(1), np.ones(1))
    policy = ACTLowDimPolicy(model, norm, mode="latest_first_action")
    outputs = [float(policy(np.array([0.0]))[0]) for _ in range(3)]
    np.testing.assert_allclose(outputs, [0.1, 0.2, 0.3])
    assert model.query_count == 3


def test_temporal_ensemble_aligns_and_clears_episode_history() -> None:
    model = CountingACT()
    norm = ObservationNormalization(np.zeros(1), np.ones(1))
    policy = ACTLowDimPolicy(model, norm, mode="temporal_ensemble", decay=0.0)
    outputs = [float(policy(np.array([0.0]))[0]) for _ in range(3)]
    np.testing.assert_allclose(outputs, [0.1, 0.3, 0.5])
    assert model.query_count == 3
    policy.reset()
    assert float(policy(np.array([0.0]))[0]) == pytest.approx(0.4)
    assert sorted(policy._chunks) == [0]


def test_checkpoint_reload_keeps_normalization_and_zero_latent_output(tmp_path) -> None:
    torch.manual_seed(5)
    model = ACTLowDim(2, 1, 3, hidden_size=8, latent_size=2, attention_heads=2)
    norm = ObservationNormalization(
        np.array([1.0, -2.0], dtype=np.float32),
        np.array([2.0, 4.0], dtype=np.float32),
    )
    path = tmp_path / "act.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": {
                "observation_size": 2,
                "action_size": 1,
                "horizon": 3,
                "hidden_size": 8,
                "latent_size": 2,
                "attention_heads": 2,
            },
            "model_state_dict": model.state_dict(),
            "observation_mean": torch.from_numpy(norm.mean),
            "observation_scale": torch.from_numpy(norm.scale),
            "source_dataset_sha256": "teaching-fixture",
            "seed": 5,
        },
        path,
    )
    restored, restored_norm, _ = load_act_lowdim_checkpoint(path)
    np.testing.assert_array_equal(restored_norm.mean, norm.mean)
    np.testing.assert_array_equal(restored_norm.scale, norm.scale)
    observation = torch.tensor([[0.0, 0.5]])
    with torch.inference_mode():
        torch.testing.assert_close(
            model.eval()(observation).actions, restored(observation).actions
        )
    assert restored(observation).posterior_mean is None
