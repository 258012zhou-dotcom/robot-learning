"""Reproduce both SAC arms and inspect all held-out closed-loop trajectories."""

import csv
import json
import math

import numpy as np
import torch

from robot_learning.gymnasium_rollout import EpisodeResult, run_episode
from robot_learning.reach_trajectory_analysis import summarize_reach_trajectory
from robot_learning.sac_direction_schedule import ScheduledTargetEnvironment
from robot_learning.sac_evaluation import DeterministicSACPolicy
from robot_learning.sac_models import SquashedGaussianActor, TwinQCritic
from robot_learning.sac_online import OnlineReplayBuffer, run_online_steps
from robot_learning.sac_training import make_target_critic

from run import CONFIG_PATH, OUTPUT_DIR, make_environment


CASE_SEEDS = (200026, 200027, 200029)
REFERENCE_FIELDS = (
    "target_position", "total_reward", "final_distance",
    "final_position", "final_velocity",
)


def reproduce_actor(config: dict, *, balanced: bool, reference: dict) -> SquashedGaussianActor:
    """Repeat the same training, refusing a changed training summary."""
    seed = int(config["training_seed"])
    torch.manual_seed(seed)
    actor = SquashedGaussianActor()
    critic = TwinQCritic()
    environment = ScheduledTargetEnvironment(
        make_environment(config, max_episode_steps=int(config["training_episode_steps"])),
        target_magnitudes=tuple(float(value) for value in config["target_magnitudes"]),
        alternate_direction=balanced,
    )
    try:
        training = run_online_steps(
            environment, actor, critic, make_target_critic(critic),
            torch.optim.Adam(actor.parameters(), lr=float(config["learning_rate"])),
            torch.optim.Adam(critic.parameters(), lr=float(config["learning_rate"])),
            OnlineReplayBuffer(int(config["replay_capacity"])),
            steps=int(config["training_steps"]),
            random_steps=int(config["random_steps"]),
            batch_size=int(config["batch_size"]),
            seed=seed,
            rng=np.random.default_rng(seed),
            gamma=float(config["gamma"]),
            alpha=float(config["alpha"]),
            tau=float(config["tau"]),
        )
    finally:
        environment.close()
    if len(environment.scheduled_targets) != len(config["target_magnitudes"]):
        raise RuntimeError("the reproduced episode schedule changed")
    if not all(math.isclose(
        float(getattr(training, key)), float(value), rel_tol=1e-8, abs_tol=1e-8,
    ) for key, value in reference.items()):
        raise RuntimeError("the reproduced training summary changed")
    return actor


def check_episode(episode: EpisodeResult, saved: dict[str, str], *, policy: str, seed: int) -> None:
    """Match each replay to the original held-out evaluation record."""
    if saved["policy"] != policy or int(saved["seed"]) != seed:
        raise RuntimeError("the saved evaluation row does not match this episode")
    if saved["success"] != str(episode.is_success) or int(saved["steps"]) != episode.step_count:
        raise RuntimeError(f"replay outcome changed: {policy}, {seed}")
    actual = {
        "target_position": float(episode.observations[0, 2]),
        "total_reward": episode.total_reward,
        "final_distance": abs(float(episode.observations[-1, 3])),
        "final_position": float(episode.observations[-1, 0]),
        "final_velocity": float(episode.observations[-1, 1]),
    }
    if any(not math.isclose(
        float(saved[field]), actual[field], rel_tol=1e-8, abs_tol=1e-6,
    ) for field in REFERENCE_FIELDS):
        raise RuntimeError(f"replay trajectory changed: {policy}, {seed}")


def trace_rows(episode: EpisodeResult, *, policy: str, seed: int) -> list[dict]:
    """Save stepwise evidence only for the three explanatory cases."""
    return [{
        "policy": policy,
        "seed": seed,
        "step": step + 1,
        "target": float(episode.observations[step, 2]),
        "position_before": float(episode.observations[step, 0]),
        "velocity_before": float(episode.observations[step, 1]),
        "action": float(episode.actions[step, 0]),
        "position_after": float(episode.observations[step + 1, 0]),
        "velocity_after": float(episode.observations[step + 1, 1]),
        "reward": float(episode.rewards[step]),
    } for step in range(episode.step_count)]


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    with (OUTPUT_DIR / "results.json").open(encoding="utf-8") as file:
        original = json.load(file)
    if original["protocol"] != config:
        raise RuntimeError("saved result protocol differs from current configuration")
    with (OUTPUT_DIR / "episodes.csv").open(encoding="utf-8", newline="") as file:
        saved_rows = {
            (row["policy"], int(row["seed"])): row for row in csv.DictReader(file)
        }
    seeds = range(
        int(config["evaluation_seed"]),
        int(config["evaluation_seed"]) + int(config["evaluation_episode_count"]),
    )
    if len(saved_rows) != 3 * len(seeds) or any(seed not in seeds for seed in CASE_SEEDS):
        raise RuntimeError("saved evaluation is incomplete or example seed is not held out")

    torch.set_num_threads(1)
    actors = {
        policy: reproduce_actor(
            config, balanced=(policy == "balanced"),
            reference=original[policy]["training"],
        )
        for policy in ("positive_only", "balanced")
    }
    environment = make_environment(
        config, max_episode_steps=int(config["evaluation_episode_steps"]),
    )
    summaries: list[dict] = []
    selected_steps: list[dict] = []
    try:
        for policy, actor in actors.items():
            deterministic_policy = DeterministicSACPolicy(actor)
            for seed in seeds:
                episode = run_episode(environment, deterministic_policy, seed=seed)
                check_episode(episode, saved_rows[(policy, seed)], policy=policy, seed=seed)
                diagnostics = summarize_reach_trajectory(
                    episode,
                    position_tolerance=environment.success_tolerance,
                    velocity_tolerance=environment.velocity_tolerance,
                )
                summaries.append({
                    "policy": policy,
                    "seed": seed,
                    "target": float(episode.observations[0, 2]),
                    "success": episode.is_success,
                    "steps": episode.step_count,
                    "total_reward": episode.total_reward,
                    "initial_action": float(episode.actions[0, 0]),
                    "closest_distance": float(np.min(np.abs(episode.observations[:, 3]))),
                    **diagnostics,
                })
                if seed in CASE_SEEDS:
                    selected_steps.extend(trace_rows(episode, policy=policy, seed=seed))
    finally:
        environment.close()

    counts = {}
    for policy in actors:
        records = [item for item in summaries if item["policy"] == policy]
        counts[policy] = {
            "episodes": len(records),
            "success": sum(item["success"] for item in records),
            "never_near_position": sum(item["first_near_step"] is None for item in records),
            "near_position_but_not_success": sum(
                item["first_near_step"] is not None and not item["success"]
                for item in records
            ),
            "crossed_target": sum(item["first_crossing_step"] is not None for item in records),
            "ever_opposed_target_direction": sum(
                item["first_opposing_action_step"] is not None for item in records
            ),
        }
    report = {
        "replay_verified": True,
        "case_seeds": list(CASE_SEEDS),
        "counts": counts,
        "summaries": summaries,
    }
    with (OUTPUT_DIR / "trajectory_diagnosis.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "trajectory_steps.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(selected_steps[0]))
        writer.writeheader()
        writer.writerows(selected_steps)
    print(json.dumps({"counts": counts, "cases": [
        item for item in summaries if item["seed"] in CASE_SEEDS
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
