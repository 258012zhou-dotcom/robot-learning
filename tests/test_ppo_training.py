"""Unit and small MuJoCo integration tests for the on-policy PPO data flow."""

from pathlib import Path

import numpy as np
import pytest
import torch

from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.ppo_training import (
    ACTION_VALUES,
    DiscreteActorCritic,
    OnPolicyBatch,
    collect_episodes,
    generalized_advantages,
    near_target_speed_cost,
    update_from_batch,
)


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "017_mujoco_step"
    / "point_robot.xml"
)


@pytest.mark.parametrize(
    ("terminated", "expected_targets"),
    [(True, [3.0, 2.0]), (False, [8.0, 7.0])],
)
def test_gae_uses_terminal_zero_or_truncation_bootstrap(
    terminated: bool, expected_targets: list[float]
) -> None:
    values = torch.tensor([0.4, 0.5])
    advantages, targets = generalized_advantages(
        torch.tensor([1.0, 2.0]),
        values,
        torch.tensor(5.0),
        terminated=terminated,
        gamma=1.0,
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(targets.numpy(), expected_targets)
    np.testing.assert_allclose(advantages.numpy(), np.asarray(expected_targets) - values.numpy())


def test_rollout_stores_old_probabilities_and_valid_actions() -> None:
    environment = PointRobotReachEnv(MODEL_PATH, max_episode_steps=3)
    model = DiscreteActorCritic()
    torch.manual_seed(37)
    batch = collect_episodes(
        environment,
        model,
        episode_seeds=range(37, 39),
        gamma=0.99,
        gae_lambda=0.95,
        reward_scale=0.01,
    )
    environment.close()

    assert len(batch.episode_records) == 2
    assert batch.observations.shape == (6, 4)
    assert batch.actions.shape == batch.old_log_probabilities.shape
    assert set(batch.actions.tolist()).issubset(set(range(len(ACTION_VALUES))))
    assert all(record["truncated"] for record in batch.episode_records)
    with torch.no_grad():
        expected_log_probabilities = model.distribution(batch.observations).log_prob(
            batch.actions
        )
    torch.testing.assert_close(batch.old_log_probabilities, expected_log_probabilities)
    assert not batch.old_log_probabilities.requires_grad
    assert torch.isfinite(batch.value_targets).all()


def test_update_changes_model_but_not_old_batch_targets() -> None:
    torch.manual_seed(38)
    model = DiscreteActorCritic()
    observations = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, -1.0, -1.0]])
    actions = torch.tensor([2, 0])
    with torch.no_grad():
        old_log_probabilities = model.distribution(observations).log_prob(actions)
    batch = OnPolicyBatch(
        observations=observations,
        actions=actions,
        old_log_probabilities=old_log_probabilities.clone(),
        advantages=torch.tensor([1.0, -1.0]),
        value_targets=torch.tensor([1.0, -1.0]),
        episode_records=[],
    )
    original_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    metrics = update_from_batch(
        model,
        optimizer,
        batch,
        epochs=2,
        clip_epsilon=0.2,
        value_loss_weight=0.5,
        entropy_weight=0.0,
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(original_parameters, model.parameters())
    )
    torch.testing.assert_close(batch.old_log_probabilities, old_log_probabilities)
    assert set(metrics) == {"policy_loss", "value_loss", "entropy"}


def test_empty_gae_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal non-empty"):
        generalized_advantages(
            torch.tensor([]), torch.tensor([]), torch.tensor(0.0),
            terminated=True, gamma=0.99, gae_lambda=0.95,
        )


def test_near_target_speed_cost_only_acts_near_target() -> None:
    assert near_target_speed_cost(
        np.asarray([0.0, 2.0, 1.0, 1.0]), radius=0.5
    ) == 0.0
    assert near_target_speed_cost(
        np.asarray([0.0, -2.0, 0.1, 0.1]), radius=0.5
    ) == pytest.approx(1.6)
    assert near_target_speed_cost(
        np.asarray([0.0, 0.0, 0.1, 0.1]), radius=0.5
    ) == 0.0


def test_zero_penalty_preserves_original_rollout_data() -> None:
    environment = PointRobotReachEnv(MODEL_PATH, max_episode_steps=3)
    model = DiscreteActorCritic()
    arguments = dict(
        episode_seeds=range(50, 52), gamma=0.99,
        gae_lambda=0.95, reward_scale=0.01,
    )
    torch.manual_seed(50)
    original = collect_episodes(environment, model, **arguments)
    torch.manual_seed(50)
    explicit_zero = collect_episodes(
        environment, model, **arguments,
        velocity_penalty_weight=0.0, near_target_radius=0.5,
    )
    environment.close()

    torch.testing.assert_close(original.actions, explicit_zero.actions)
    torch.testing.assert_close(original.old_log_probabilities, explicit_zero.old_log_probabilities)
    torch.testing.assert_close(original.value_targets, explicit_zero.value_targets)
    assert all(record["total_speed_penalty"] == 0.0 for record in explicit_zero.episode_records)


def test_speed_penalty_changes_training_target_not_raw_episode_reward() -> None:
    class OneStepNearTargetEnv:
        def reset(self, *, seed: int) -> tuple[np.ndarray, dict]:
            del seed
            return np.asarray([0.0, 0.0, 0.1, 0.1], dtype=np.float32), {}

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
            del action
            next_observation = np.asarray([0.05, 1.0, 0.1, 0.05], dtype=np.float32)
            return next_observation, 1.0, True, False, {
                "target_position": 0.1, "is_success": False
            }

    environment = OneStepNearTargetEnv()
    model = DiscreteActorCritic()
    batch = collect_episodes(
        environment, model, episode_seeds=range(1, 2),
        gamma=1.0, gae_lambda=1.0, reward_scale=1.0,
        velocity_penalty_weight=0.5, near_target_radius=0.5,
    )

    assert batch.episode_records[0]["total_reward"] == 1.0
    assert batch.episode_records[0]["total_speed_penalty"] == pytest.approx(0.45)
    assert batch.value_targets.item() == pytest.approx(0.55)
