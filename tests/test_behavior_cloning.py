"""Unit tests for behavior-cloning data selection and normalization."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from robot_learning.behavior_cloning import (
    BehaviorCloningMLP,
    BehaviorCloningPolicy,
    ObservationNormalization,
    fit_observation_normalization,
    load_behavior_cloning_checkpoint,
    prepare_behavior_cloning_data,
    save_behavior_cloning_checkpoint,
)
from robot_learning.gymnasium_rollout import EpisodeResult, run_episode
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.training import create_data_loader, evaluate_mse, train_model
from robot_learning.trajectory_dataset import (
    CollectedEpisode,
    TEST_SPLIT_ID,
    TRAIN_SPLIT_ID,
    VALIDATION_SPLIT_ID,
    TransitionDataset,
    build_transition_dataset,
)


def make_result(base: float, action: float) -> EpisodeResult:
    """Create two deterministic transitions with four varying features."""
    observations = np.asarray(
        [
            [base, base + 1.0, base + 2.0, base + 3.0],
            [base + 1.0, base + 3.0, base + 5.0, base + 7.0],
            [base + 2.0, base + 5.0, base + 8.0, base + 11.0],
        ],
        dtype=np.float32,
    )
    actions = np.full((2, 1), action, dtype=np.float32)
    rewards = np.asarray([-1.0, -0.5], dtype=np.float64)
    return EpisodeResult(
        observations=observations,
        actions=actions,
        rewards=rewards,
        total_reward=float(rewards.sum()),
        terminated=True,
        truncated=False,
        is_success=True,
    )


def make_dataset() -> TransitionDataset:
    """Return random data plus one expert Episode in every split."""
    return build_transition_dataset(
        [
            CollectedEpisode(
                0, 10, 0, TRAIN_SPLIT_ID, make_result(100.0, -0.75)
            ),
            CollectedEpisode(
                1, 11, 1, TRAIN_SPLIT_ID, make_result(0.0, 0.25)
            ),
            CollectedEpisode(
                2, 12, 1, VALIDATION_SPLIT_ID, make_result(50.0, 0.50)
            ),
            CollectedEpisode(
                3, 13, 1, TEST_SPLIT_ID, make_result(-50.0, -0.25)
            ),
        ]
    )


def test_preparation_selects_only_expert_rows_and_preserves_splits() -> None:
    """Random actions must not become labels and Episodes must not leak."""
    prepared = prepare_behavior_cloning_data(make_dataset(), expert_policy_id=1)

    np.testing.assert_array_equal(prepared.train.actions, [[0.25], [0.25]])
    np.testing.assert_array_equal(
        prepared.validation.actions, [[0.50], [0.50]]
    )
    np.testing.assert_array_equal(prepared.test.actions, [[-0.25], [-0.25]])
    assert set(prepared.train.episode_ids.tolist()) == {1}
    assert set(prepared.validation.episode_ids.tolist()) == {2}
    assert set(prepared.test.episode_ids.tolist()) == {3}


def test_normalization_statistics_use_training_expert_rows_only() -> None:
    """Validation, test, and random rows must not affect fitted statistics."""
    dataset = make_dataset()
    expert_train_rows = (dataset.policy_ids == 1) & (
        dataset.split_ids == TRAIN_SPLIT_ID
    )
    expected_observations = dataset.observations[expert_train_rows]

    prepared = prepare_behavior_cloning_data(dataset, expert_policy_id=1)

    np.testing.assert_allclose(
        prepared.observation_normalization.mean,
        expected_observations.mean(axis=0),
    )
    np.testing.assert_allclose(
        prepared.observation_normalization.scale,
        expected_observations.std(axis=0),
    )
    np.testing.assert_allclose(prepared.train.observations.mean(axis=0), 0.0)
    np.testing.assert_allclose(prepared.train.observations.std(axis=0), 1.0)


def test_constant_feature_uses_unit_scale_instead_of_dividing_by_zero() -> None:
    """A constant input feature should normalize to zero without NaNs."""
    observations = np.asarray([[1.0, 2.0], [1.0, 4.0]], dtype=np.float32)

    normalization = fit_observation_normalization(observations)
    normalized = normalization.transform(observations)

    np.testing.assert_array_equal(normalization.scale, [1.0, 1.0])
    np.testing.assert_array_equal(normalized[:, 0], [0.0, 0.0])
    assert np.all(np.isfinite(normalized))


def test_missing_expert_split_is_rejected() -> None:
    """Training must not silently continue without an evaluation split."""
    dataset = make_dataset()
    changed_split_ids = dataset.split_ids.copy()
    changed_split_ids[dataset.episode_ids == 3] = VALIDATION_SPLIT_ID
    dataset_without_test = replace(dataset, split_ids=changed_split_ids)

    with pytest.raises(ValueError, match="has no rows in split"):
        prepare_behavior_cloning_data(dataset_without_test, expert_policy_id=1)


def test_mlp_returns_one_bounded_action_per_observation() -> None:
    """The policy output must match the environment action contract."""
    torch.manual_seed(7)
    model = BehaviorCloningMLP(
        observation_size=4,
        action_low=[-1.0],
        action_high=[1.0],
        hidden_size=8,
    )
    observations = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1000.0, -1000.0, 500.0, -500.0],
        ],
        dtype=torch.float32,
    )

    actions = model(observations)

    assert actions.shape == (2, 1)
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)


def test_mlp_scales_tanh_output_to_asymmetric_action_bounds() -> None:
    """Bounds should work for action spaces that are not centered on zero."""
    model = BehaviorCloningMLP(4, action_low=[-2.0], action_high=[4.0])
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    action = model(torch.zeros((1, 4), dtype=torch.float32))

    torch.testing.assert_close(action, torch.tensor([[1.0]]))


def test_mlp_rejects_observation_with_wrong_shape() -> None:
    """A missing batch or feature dimension must fail before environment use."""
    model = BehaviorCloningMLP(4, action_low=[-1.0], action_high=[1.0])

    with pytest.raises(ValueError, match=r"shape \(N, 4\)"):
        model(torch.zeros(4, dtype=torch.float32))


def test_mlp_rejects_invalid_action_bounds() -> None:
    """An empty, reversed, or mismatched action range is not a valid policy."""
    with pytest.raises(ValueError, match="equal-length vectors"):
        BehaviorCloningMLP(4, action_low=[-1.0], action_high=[1.0, 2.0])
    with pytest.raises(ValueError, match="below"):
        BehaviorCloningMLP(4, action_low=[1.0], action_high=[-1.0])


def test_mlp_learns_simple_expert_mapping_and_restores_best_model() -> None:
    """The bounded BC model must work with the shared training pipeline."""
    torch.manual_seed(19)
    feature = torch.linspace(-1.0, 1.0, 120).reshape(-1, 1)
    observations = torch.cat(
        [feature, feature**2, -feature, torch.ones_like(feature)], dim=1
    )
    actions = 0.5 * feature
    train_loader = create_data_loader(
        observations[:90],
        actions[:90],
        batch_size=30,
        shuffle=True,
        seed=23,
    )
    validation_loader = create_data_loader(
        observations[90:],
        actions[90:],
        batch_size=30,
        shuffle=False,
        seed=23,
    )
    model = BehaviorCloningMLP(4, action_low=[-1.0], action_high=[1.0])
    device = torch.device("cpu")
    initial_mse = evaluate_mse(model, validation_loader, device)

    result = train_model(
        model,
        train_loader,
        validation_loader,
        device=device,
        epochs=80,
        learning_rate=0.01,
        weight_decay=0.0,
    )
    restored_mse = evaluate_mse(model, validation_loader, device)

    assert result.best_validation_loss < initial_mse * 0.05
    assert restored_mse == pytest.approx(result.best_validation_loss)


def test_checkpoint_round_trip_preserves_weights_normalization_and_output(
    tmp_path,
) -> None:
    """A fresh model reconstructed from disk must make identical predictions."""
    torch.manual_seed(31)
    model = BehaviorCloningMLP(4, action_low=[-1.0], action_high=[1.0])
    normalization = ObservationNormalization(
        mean=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        scale=np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
    )
    raw_observations = np.asarray(
        [[1.5, 1.0, 3.0, 8.0], [-1.0, 2.5, 6.0, 0.0]],
        dtype=np.float32,
    )
    normalized = torch.from_numpy(normalization.transform(raw_observations))
    with torch.inference_mode():
        expected_actions = model(normalized)
    path = tmp_path / "bc.pt"
    save_behavior_cloning_checkpoint(
        path,
        model,
        normalization,
        source_dataset_sha256="abc123",
        seed=31,
    )

    torch.manual_seed(999)
    loaded, loaded_normalization, artifact = (
        load_behavior_cloning_checkpoint(path)
    )
    with torch.inference_mode():
        actual_actions = loaded(
            torch.from_numpy(
                loaded_normalization.transform(raw_observations)
            )
        )

    assert not loaded.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    np.testing.assert_array_equal(loaded_normalization.mean, normalization.mean)
    np.testing.assert_array_equal(
        loaded_normalization.scale, normalization.scale
    )
    torch.testing.assert_close(actual_actions, expected_actions, rtol=0, atol=0)
    assert artifact["source_dataset_sha256"] == "abc123"


def test_loaded_policy_applies_normalization_to_one_raw_observation(
    tmp_path,
) -> None:
    """The rollout adapter must reproduce direct normalized model inference."""
    model = BehaviorCloningMLP(4, action_low=[-1.0], action_high=[1.0])
    normalization = ObservationNormalization(
        mean=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        scale=np.asarray([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )
    path = tmp_path / "bc.pt"
    save_behavior_cloning_checkpoint(
        path,
        model,
        normalization,
        source_dataset_sha256="abc123",
        seed=7,
    )
    loaded, loaded_normalization, _ = load_behavior_cloning_checkpoint(path)
    policy = BehaviorCloningPolicy(loaded, loaded_normalization)
    observation = np.asarray([2.0, 4.0, 6.0, 8.0], dtype=np.float32)

    action = policy(observation)
    with torch.inference_mode():
        expected = loaded(
            torch.from_numpy(normalization.transform(observation[None, :]))
        )[0].numpy()

    assert action.shape == (1,)
    np.testing.assert_array_equal(action, expected)


def test_checkpoint_rejects_unknown_format_version(tmp_path) -> None:
    """Loading must fail instead of guessing how an unknown artifact works."""
    path = tmp_path / "future.pt"
    torch.save({"format_version": 99}, path)

    with pytest.raises(ValueError, match="format_version"):
        load_behavior_cloning_checkpoint(path)


def test_behavior_cloning_policy_runs_inside_real_environment() -> None:
    """The inference adapter must satisfy the real rollout action contract."""
    root = Path(__file__).resolve().parents[1]
    environment = PointRobotReachEnv(
        root / "experiments/017_mujoco_step/point_robot.xml",
        max_episode_steps=3,
    )
    model = BehaviorCloningMLP(4, action_low=[-1.0], action_high=[1.0])
    normalization = ObservationNormalization(
        mean=np.zeros(4, dtype=np.float32),
        scale=np.ones(4, dtype=np.float32),
    )
    policy = BehaviorCloningPolicy(model, normalization)
    try:
        result = run_episode(environment, policy, seed=1234)
    finally:
        environment.close()

    assert result.actions.shape == (3, 1)
    assert np.all(result.actions >= -1.0)
    assert np.all(result.actions <= 1.0)
