"""Independent, deterministic closed-loop evaluation for the SAC learner."""

import numpy as np
import torch

from robot_learning.gymnasium_rollout import (
    Policy,
    PolicyEvaluation,
    run_episode,
    summarize_episodes,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.sac_models import SquashedGaussianActor


class DeterministicSACPolicy:
    """Use the bounded Gaussian mean rather than sampling during evaluation."""

    def __init__(self, actor: SquashedGaussianActor) -> None:
        self.actor = actor

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        device = next(self.actor.parameters()).device
        with torch.no_grad():
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device,
            ).unsqueeze(0)
            action = self.actor.deterministic_action(observation_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32)


def evaluate_reach_policy(
    environment: PointRobotReachEnv,
    policy: Policy,
    seeds: range,
) -> tuple[PolicyEvaluation, list[dict[str, int | float | bool]]]:
    """Evaluate complete held-out Episodes and retain each final state."""
    if not seeds:
        raise ValueError("evaluation seeds must not be empty")
    episodes = [run_episode(environment, policy, seed=seed) for seed in seeds]
    records: list[dict[str, int | float | bool]] = []
    for seed, episode in zip(seeds, episodes, strict=True):
        target = float(episode.observations[0, 2])
        final_position = float(episode.observations[-1, 0])
        final_velocity = float(episode.observations[-1, 1])
        records.append({
            "seed": seed,
            "target_position": target,
            "success": episode.is_success,
            "terminated": episode.terminated,
            "truncated": episode.truncated,
            "steps": episode.step_count,
            "total_reward": episode.total_reward,
            "final_distance": abs(float(episode.observations[-1, 3])),
            "final_position": final_position,
            "final_velocity": final_velocity,
            "end_beyond_target": target * (final_position - target) > 0.0,
        })
    return summarize_episodes(episodes), records
