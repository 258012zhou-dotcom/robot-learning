"""Check SAC's bounded replay and online collect/update sequence."""

import gymnasium as gym
import numpy as np
import torch

from robot_learning.sac_models import SquashedGaussianActor, TwinQCritic
from robot_learning.sac_online import OnlineReplayBuffer, run_online_steps
from robot_learning.sac_training import make_target_critic


class TwoStepEnvironment(gym.Env):
    """Tiny deterministic environment for checking interaction bookkeeping."""

    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
        self.seeds: list[int | None] = []
        self.actions: list[np.ndarray] = []
        self.episode_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seeds.append(seed)
        self.episode_step = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.episode_step += 1
        observation = np.full(4, self.episode_step, dtype=np.float32)
        done = self.episode_step == 2
        # One true termination, then a time-limit truncation.
        terminated = done and len(self.seeds) == 1
        truncated = done and len(self.seeds) > 1
        return observation, -1.0, terminated, truncated, {}


def test_replay_preserves_transition_values_and_discards_oldest() -> None:
    replay = OnlineReplayBuffer(capacity=2)
    observation = np.zeros(4, dtype=np.float32)
    for index in range(3):
        replay.append(observation, np.asarray([0.5]), -float(index),
                      np.full(4, index + 1), index == 1, index == 2)
    observation[:] = 99.0

    assert len(replay) == 2
    assert [item.reward for item in replay.transitions] == [-1.0, -2.0]
    assert replay.transitions[0].terminated and replay.transitions[1].truncated
    sampled = replay.sample(20, np.random.default_rng(3))
    assert sampled.actions.shape == (20, 1)
    assert set(sampled.rewards.tolist()) <= {-1.0, -2.0}
    assert np.all(sampled.observations == 0.0)


def test_online_loop_counts_steps_updates_and_episode_boundaries() -> None:
    torch.manual_seed(11)
    environment = TwoStepEnvironment()
    actor = SquashedGaussianActor()
    critic = TwinQCritic()
    replay = OnlineReplayBuffer(capacity=4)
    summary = run_online_steps(
        environment, actor, critic, make_target_critic(critic),
        torch.optim.Adam(actor.parameters(), lr=1e-3),
        torch.optim.Adam(critic.parameters(), lr=1e-3),
        replay, steps=5, random_steps=2, batch_size=2, seed=40,
        rng=np.random.default_rng(40), gamma=0.99, alpha=0.2, tau=0.005,
    )

    assert summary.environment_steps == 5
    assert summary.gradient_updates == 3
    assert summary.completed_episodes == 2
    assert summary.replay_size == 4
    assert environment.seeds == [40, 41, 42]
    assert len(environment.actions) == 5
    assert all(action.shape == (1,) and np.abs(action[0]) <= 1.0
               for action in environment.actions)
    assert replay.transitions[0].terminated
    assert replay.transitions[2].truncated
    assert np.isfinite(summary.last_critic_loss)
    assert np.isfinite(summary.last_actor_loss)
