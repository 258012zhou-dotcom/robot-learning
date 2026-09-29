"""Unit checks for route labels, split isolation, and route-side evaluation."""

import numpy as np
import pytest
import torch

from robot_learning.behavior_cloning import ObservationNormalization
from robot_learning.gymnasium_rollout import EpisodeResult, run_episode
from robot_learning.route_conditioning import (
    RouteConditionedSequenceBCPolicy,
    prepare_route_conditioned_data,
    route_at_obstacle_x,
)
from robot_learning.sequence_behavior_cloning import SequenceBCMLP
from robot_learning.trajectory_dataset import (
    CollectedEpisode,
    TEST_SPLIT_ID,
    TRAIN_SPLIT_ID,
    TransitionDataset,
    VALIDATION_SPLIT_ID,
    build_transition_dataset,
)
from robot_learning.two_route_navigation import TwoRouteExpert, TwoRouteNavigationEnv


def make_dataset(*, leak_scene: bool = False) -> TransitionDataset:
    episodes = []
    for split_index, (split_id, value) in enumerate((
        (TRAIN_SPLIT_ID, 2.0),
        (VALIDATION_SPLIT_ID, 100.0),
        (TEST_SPLIT_ID, 200.0),
    )):
        scene_seed = 10 if leak_scene and split_index == 1 else 10 + split_index
        for policy_id, route in ((0, -1), (1, 1)):
            observation = np.array([[value], [value + 1]], dtype=np.float32)
            result = EpisodeResult(
                observations=observation,
                actions=np.array([[float(route)]], dtype=np.float32),
                rewards=np.zeros(1),
                total_reward=0.0,
                terminated=True,
                truncated=False,
                is_success=True,
            )
            episodes.append(CollectedEpisode(
                episode_id=2 * split_index + policy_id,
                environment_seed=scene_seed,
                policy_id=policy_id,
                split_id=split_id,
                result=result,
            ))
    return build_transition_dataset(episodes)


def test_conditioned_data_keeps_both_routes_and_train_only_normalization() -> None:
    prepared = prepare_route_conditioned_data(make_dataset(), horizon=2)
    np.testing.assert_array_equal(prepared.normalization.mean, [2.0, 0.0])
    np.testing.assert_array_equal(prepared.normalization.scale, [1.0, 1.0])
    np.testing.assert_array_equal(prepared.train.observations, [[0.0, -1.0], [0.0, 1.0]])
    np.testing.assert_array_equal(prepared.validation.observations[:, 0], [98.0, 98.0])
    np.testing.assert_array_equal(prepared.test.episode_ids, [4, 5])
    with pytest.raises(ValueError, match="scene leakage"):
        prepare_route_conditioned_data(make_dataset(leak_scene=True), horizon=2)


def test_policy_receives_fixed_route_instruction_and_resets_chunk(monkeypatch) -> None:
    model = SequenceBCMLP(8, [-1.0, -1.0], [1.0, 1.0], horizon=2, hidden_size=8)
    seen = []

    def fake_forward(observations: torch.Tensor) -> torch.Tensor:
        seen.append(observations.detach().clone())
        direction = float(observations[0, -1])
        return torch.tensor([[[0.0, direction], [0.0, direction]]])

    monkeypatch.setattr(model, "forward", fake_forward)
    normalization = ObservationNormalization(np.zeros(8), np.ones(8))
    policy = RouteConditionedSequenceBCPolicy(model, normalization, execution_horizon=2)
    with pytest.raises(RuntimeError, match="route instruction"):
        policy(np.zeros(7))
    policy.reset(route=1)
    np.testing.assert_allclose(policy(np.zeros(7)), [0.0, 1.0])
    policy.reset(route=-1)
    np.testing.assert_allclose(policy(np.zeros(7)), [0.0, -1.0])
    assert [float(value[0, -1]) for value in seen] == [1.0, -1.0]
    with pytest.raises(ValueError, match="route instruction"):
        policy.reset(route=0)


@pytest.mark.parametrize("route", [-1, 1])
def test_route_metric_accepts_successful_expert_side(route: int) -> None:
    environment = TwoRouteNavigationEnv()
    result = run_episode(environment, TwoRouteExpert(route), seed=7)
    assert result.is_success
    assert route_at_obstacle_x(result.observations) == route
    environment.close()


def test_route_metric_rejects_center_and_uncrossed_paths() -> None:
    base = np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.4])
    through_center = np.stack((base, np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.4])))
    assert route_at_obstacle_x(through_center) == 0
    stopped_left = np.stack((base, np.array([-0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.4])))
    assert route_at_obstacle_x(stopped_left) is None
