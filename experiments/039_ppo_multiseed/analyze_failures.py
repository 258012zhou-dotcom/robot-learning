"""Replay fixed multi-seed PPO Episodes without retraining or changing evaluation."""

import csv
import json
import math

import numpy as np
import torch

from robot_learning.gymnasium_rollout import (
    EpisodeResult,
    ProportionalReachPolicy,
    run_episode,
)
from robot_learning.ppo_training import DiscreteActorCritic
from robot_learning.reach_trajectory_analysis import summarize_reach_trajectory

from run import CONFIG_PATH, OUTPUT_DIR, GreedyDiscretePolicy, make_environment


def trace_rows(
    episode: EpisodeResult, *, policy_name: str, seed: int
) -> list[dict[str, float | int | str]]:
    """Record each action with its observation before and after the step."""
    rows = []
    for step in range(episode.step_count):
        before = episode.observations[step]
        after = episode.observations[step + 1]
        rows.append({
            "policy": policy_name,
            "seed": seed,
            "step": step,
            "target": float(before[2]),
            "position_before": float(before[0]),
            "velocity_before": float(before[1]),
            "action": float(episode.actions[step, 0]),
            "position_after": float(after[0]),
            "velocity_after": float(after[1]),
            "reward": float(episode.rewards[step]),
        })
    return rows


def check_replay(
    episode: EpisodeResult, *, policy_name: str, seed: int,
    reference: dict[tuple[str, str, int], dict[str, str]],
) -> None:
    """Refuse to interpret a replay that differs from the saved evaluation."""
    training_seed = policy_name.removeprefix("ppo_") if policy_name != "proportional" else ""
    policy = "trained_greedy" if training_seed else "proportional"
    saved = reference.get((policy, training_seed, seed))
    last = episode.observations[-1]
    if saved is None or saved["success"] != str(episode.is_success) or int(saved["steps"]) != episode.step_count:
        raise RuntimeError(f"replay does not match saved evaluation: {policy_name}, {seed}")
    for field, actual in (
        ("target_position", float(episode.observations[0, 2])),
        ("total_reward", episode.total_reward),
        ("final_distance", abs(float(last[3]))),
        ("final_position", float(last[0])),
        ("final_velocity", float(last[1])),
    ):
        if not math.isclose(float(saved[field]), actual, rel_tol=1e-8, abs_tol=1e-6):
            raise RuntimeError(f"replay changed {field}: {policy_name}, {seed}")


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        experiment = json.load(file)
    with (CONFIG_PATH.parents[1] / experiment["base_config"]).open(encoding="utf-8") as file:
        base = json.load(file)
    seeds = [int(seed) for seed in experiment["analysis_seeds"]]
    evaluation_seeds = range(
        int(base["evaluation_seed"]),
        int(base["evaluation_seed"]) + int(base["evaluation_episode_count"]),
    )
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in evaluation_seeds for seed in seeds):
        raise ValueError("analysis seeds must be distinct held-out evaluation seeds")

    with (OUTPUT_DIR / "evaluation_episodes.csv").open(encoding="utf-8", newline="") as file:
        reference = {
            (row["policy"], row["training_seed"], int(row["seed"])): row
            for row in csv.DictReader(file)
        }
    torch.set_num_threads(1)
    environment = make_environment(base)
    policies = {}
    for training_seed in experiment["training_seeds"]:
        model = DiscreteActorCritic()
        model.load_state_dict(torch.load(
            OUTPUT_DIR / f"trained_seed_{training_seed}.pt",
            map_location="cpu", weights_only=True,
        ))
        model.eval()
        policies[f"ppo_{training_seed}"] = GreedyDiscretePolicy(model)
    policies["proportional"] = ProportionalReachPolicy(
        float(base["proportional_gain"]), environment.action_space
    )

    summaries = []
    rows = []
    for seed in seeds:
        targets = []
        for name, policy in policies.items():
            episode = run_episode(environment, policy, seed=seed)
            check_replay(episode, policy_name=name, seed=seed, reference=reference)
            target = float(episode.observations[0, 2])
            targets.append(target)
            rows.extend(trace_rows(episode, policy_name=name, seed=seed))
            summaries.append({
                "policy": name,
                "seed": seed,
                "target": target,
                "success": episode.is_success,
                "steps": episode.step_count,
                "total_reward": episode.total_reward,
                **summarize_reach_trajectory(
                    episode,
                    position_tolerance=environment.success_tolerance,
                    velocity_tolerance=environment.velocity_tolerance,
                ),
            })
        if not np.allclose(targets, targets[0]):
            raise RuntimeError(f"policies received different targets for seed {seed}")
    environment.close()

    with (OUTPUT_DIR / "trajectory_diagnosis.json").open("w", encoding="utf-8") as file:
        json.dump({"analysis_seeds": seeds, "replay_verified": True, "summaries": summaries},
                  file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "trajectory_steps.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for item in summaries:
        print(
            f"eval={item['seed']} {item['policy']}: success={item['success']}, "
            f"near={item['first_near_step']}, speed_near={item['speed_at_first_near']}, "
            f"opposing={item['first_opposing_action_step']}, "
            f"overshoot={item['maximum_overshoot']:.3f} m"
        )


if __name__ == "__main__":
    main()
