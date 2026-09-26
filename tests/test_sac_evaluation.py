"""Unit and small simulator checks for SAC's deterministic evaluation."""

from pathlib import Path

import numpy as np
import pytest
import torch

from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.sac_evaluation import (
    DeterministicSACPolicy,
    evaluate_reach_policy,
)
from robot_learning.sac_models import SquashedGaussianActor


MODEL_PATH = Path(__file__).resolve().parents[1] / "experiments/017_mujoco_step/point_robot.xml"


def test_deterministic_actor_action_is_repeatable_and_bounded() -> None:
    torch.manual_seed(9)
    actor = SquashedGaussianActor()
    policy = DeterministicSACPolicy(actor)
    observation = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    first = policy(observation)
    actor.sample(torch.zeros(1, 4))
    second = policy(observation)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (1,)
    assert np.all(np.abs(first) <= 1.0)


def test_evaluation_uses_complete_seeded_simulation_episodes() -> None:
    torch.manual_seed(10)
    actor = SquashedGaussianActor()
    before = [parameter.detach().clone() for parameter in actor.parameters()]
    environment = PointRobotReachEnv(MODEL_PATH, max_episode_steps=2)
    try:
        summary, records = evaluate_reach_policy(
            environment, DeterministicSACPolicy(actor), range(100, 102),
        )
    finally:
        environment.close()

    assert summary.episode_count == 2
    assert [record["seed"] for record in records] == [100, 101]
    assert all(record["terminated"] or record["truncated"] for record in records)
    assert all(record["steps"] <= 2 for record in records)
    assert all(torch.equal(previous, current) for previous, current in zip(
        before, actor.parameters(), strict=True,
    ))


def test_evaluation_rejects_empty_seed_list() -> None:
    environment = PointRobotReachEnv(MODEL_PATH, max_episode_steps=2)
    try:
        with pytest.raises(ValueError, match="seeds"):
            evaluate_reach_policy(environment, DeterministicSACPolicy(SquashedGaussianActor()), range(0))
    finally:
        environment.close()
