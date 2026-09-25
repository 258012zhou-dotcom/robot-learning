"""Inspect fixed evaluation Episodes without changing the PPO training setup."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    """Align each action with the observation before and after it."""
    rows: list[dict[str, float | int | str]] = []
    for step in range(episode.step_count):
        before = episode.observations[step]
        after = episode.observations[step + 1]
        rows.append(
            {
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
            }
        )
    return rows


def plot_comparison(
    seed: int, episodes: dict[str, EpisodeResult], target: float
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for name, episode in episodes.items():
        observation_steps = np.arange(episode.observations.shape[0])
        action_steps = np.arange(episode.actions.shape[0])
        axes[0].plot(observation_steps, episode.observations[:, 0], label=name)
        axes[1].plot(observation_steps, episode.observations[:, 1], label=name)
        axes[2].plot(action_steps, episode.actions[:, 0], label=name, alpha=0.8)
    axes[0].axhline(target, color="black", linestyle="--", label="target")
    axes[0].set_ylabel("position [m]")
    axes[1].set_ylabel("velocity [m/s]")
    axes[2].set_ylabel("motor action")
    axes[2].set_xlabel("environment step")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"Evaluation seed {seed}: trained PPO versus P controller")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"trace_{seed}.png", dpi=150)
    plt.close(figure)


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    seeds = [int(seed) for seed in config["analysis_seeds"]]
    evaluation_start = int(config["evaluation_seed"])
    evaluation_stop = evaluation_start + int(config["evaluation_episode_count"])
    if not seeds or len(seeds) != len(set(seeds)) or any(
        seed not in range(evaluation_start, evaluation_stop) for seed in seeds
    ):
        raise ValueError("analysis_seeds must be unique held-out evaluation seeds")

    torch.set_num_threads(1)
    model = DiscreteActorCritic()
    model.load_state_dict(
        torch.load(OUTPUT_DIR / "trained_model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    environment = make_environment(config)
    policies = {
        "trained_greedy": GreedyDiscretePolicy(model),
        "proportional": ProportionalReachPolicy(
            float(config["proportional_gain"]), environment.action_space
        ),
    }
    summaries: list[dict[str, float | int | bool | str | None]] = []
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        episodes = {
            name: run_episode(environment, policy, seed=seed)
            for name, policy in policies.items()
        }
        target = float(episodes["trained_greedy"].observations[0, 2])
        for name, episode in episodes.items():
            rows.extend(trace_rows(episode, policy_name=name, seed=seed))
            summaries.append(
                {
                    "policy": name,
                    "seed": seed,
                    "target": target,
                    "success": episode.is_success,
                    "steps": episode.step_count,
                    **summarize_reach_trajectory(
                        episode,
                        position_tolerance=environment.success_tolerance,
                        velocity_tolerance=environment.velocity_tolerance,
                    ),
                }
            )
        plot_comparison(seed, episodes, target)
    environment.close()

    with (OUTPUT_DIR / "trajectory_diagnosis.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "trajectory_steps.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for item in summaries:
        print(
            f"seed={item['seed']} {item['policy']}: success={item['success']}, "
            f"first_near={item['first_near_step']}, "
            f"speed_near={item['speed_at_first_near']}, "
            f"overshoot={item['maximum_overshoot']:.3f} m"
        )


if __name__ == "__main__":
    main()
