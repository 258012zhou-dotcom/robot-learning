"""One SAC gradient update from previously collected transitions."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from robot_learning.sac_dataflow import ReplayBatch, soft_q_targets
from robot_learning.sac_models import SquashedGaussianActor, TwinQCritic


@dataclass(frozen=True)
class SACUpdateMetrics:
    critic_loss: float
    actor_loss: float


def make_target_critic(critic: TwinQCritic) -> TwinQCritic:
    """Copy the online critic and keep its parameters out of optimization."""
    target = deepcopy(critic)
    target.requires_grad_(False)
    return target


def soft_update_target(
    target: TwinQCritic, online: TwinQCritic, *, tau: float,
) -> None:
    """Move target weights by tau of the gap toward online weights."""
    if not np.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be finite and in [0, 1]")
    with torch.no_grad():
        for target_parameter, online_parameter in zip(
            target.parameters(), online.parameters(), strict=True,
        ):
            target_parameter.lerp_(online_parameter, tau)


def sac_update(
    actor: SquashedGaussianActor,
    critic: TwinQCritic,
    target_critic: TwinQCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: ReplayBatch,
    *,
    gamma: float,
    alpha: float,
    tau: float,
) -> SACUpdateMetrics:
    """Update both Q networks, then the actor, then softly copy target weights.

    This teaching step uses fixed alpha and existing data; it collects no new data.
    """
    if not np.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be finite and in [0, 1]")
    device = next(actor.parameters()).device
    observations = torch.as_tensor(batch.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=device)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
    next_observations = torch.as_tensor(
        batch.next_observations, dtype=torch.float32, device=device,
    )
    terminated = torch.as_tensor(batch.terminated, dtype=torch.bool, device=device)

    with torch.no_grad():
        next_actions, next_log_probabilities = actor.sample(next_observations)
        next_q1, next_q2 = target_critic(next_observations, next_actions)
        targets = soft_q_targets(
            rewards, next_q1, next_q2, next_log_probabilities, terminated,
            gamma=gamma, alpha=alpha,
        )

    critic_optimizer.zero_grad(set_to_none=True)
    q1, q2 = critic(observations, actions)
    critic_loss = F.mse_loss(q1, targets) + F.mse_loss(q2, targets)
    critic_loss.backward()
    critic_optimizer.step()

    # Actor needs dQ/da, but must not update the Critic's own parameters.
    critic_optimizer.zero_grad(set_to_none=True)
    critic.requires_grad_(False)
    try:
        actor_optimizer.zero_grad(set_to_none=True)
        sampled_actions, log_probabilities = actor.sample(observations)
        policy_q1, policy_q2 = critic(observations, sampled_actions)
        actor_loss = (
            alpha * log_probabilities - torch.minimum(policy_q1, policy_q2)
        ).mean()
        actor_loss.backward()
        actor_optimizer.step()
    finally:
        critic.requires_grad_(True)

    soft_update_target(target_critic, critic, tau=tau)
    return SACUpdateMetrics(
        critic_loss=float(critic_loss.detach()),
        actor_loss=float(actor_loss.detach()),
    )
