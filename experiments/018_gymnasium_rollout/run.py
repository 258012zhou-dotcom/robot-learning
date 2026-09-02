"""Compare random and proportional policies on the same reach Episodes."""

import csv
from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from robot_learning.gymnasium_rollout import (
    EpisodeResult,
    ProportionalReachPolicy,
    RandomPolicy,
    summarize_episodes,
    run_episode,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "018_gymnasium_rollout.json"
MODEL_PATH = (
    PROJECT_ROOT / "experiments" / "017_mujoco_step" / "point_robot.xml"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "018_gymnasium_rollout"


def load_config() -> dict[str, Any]:
    """Load and minimally validate the experiment settings."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    if int(config["episode_count"]) <= 0:
        raise ValueError("episode_count must be positive")
    return config


def make_environment(config: dict[str, Any]) -> PointRobotReachEnv:
    """Create one environment from the shared experiment configuration."""
    return PointRobotReachEnv(
        MODEL_PATH,
        frame_skip=int(config["frame_skip"]),
        max_episode_steps=int(config["max_episode_steps"]),
        minimum_target_distance=float(config["minimum_target_distance"]),
        maximum_target_distance=float(config["maximum_target_distance"]),
        success_tolerance=float(config["success_tolerance"]),
        velocity_tolerance=float(config["velocity_tolerance"]),
        action_penalty_weight=float(config["action_penalty_weight"]),
    )


def evaluate_policies(
    config: dict[str, Any],
) -> dict[str, list[EpisodeResult]]:
    """Run both policies with identical environment seeds and target tasks."""
    environment = make_environment(config)
    first_seed = int(config["seed"])
    episode_seeds = range(
        first_seed,
        first_seed + int(config["episode_count"]),
    )

    policies = {
        "random": RandomPolicy(
            environment.action_space,
            seed=first_seed + int(config["random_policy_seed_offset"]),
        ),
        "proportional": ProportionalReachPolicy(
            float(config["proportional_gain"]),
            environment.action_space,
        ),
    }
    policy_results: dict[str, list[EpisodeResult]] = {}
    for policy_name, policy in policies.items():
        policy_results[policy_name] = [
            run_episode(environment, policy, seed=episode_seed)
            for episode_seed in episode_seeds
        ]

    # Equal seeds should generate equal initial targets for a fair comparison.
    random_targets = [result.observations[0, 2] for result in policy_results["random"]]
    proportional_targets = [
        result.observations[0, 2]
        for result in policy_results["proportional"]
    ]
    if not np.array_equal(random_targets, proportional_targets):
        raise RuntimeError("policy evaluations did not use equal target tasks")
    return policy_results


def save_episode_table(
    policy_results: dict[str, list[EpisodeResult]],
    first_seed: int,
) -> None:
    """Save per-Episode evidence instead of keeping only aggregate means."""
    with (OUTPUT_DIR / "episodes.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "policy",
                "episode_index",
                "environment_seed",
                "target_position",
                "success",
                "terminated",
                "truncated",
                "step_count",
                "total_reward",
                "final_position",
                "final_velocity",
                "final_distance",
            )
        )
        for policy_name, results in policy_results.items():
            for episode_index, result in enumerate(results):
                final_observation = result.observations[-1]
                writer.writerow(
                    (
                        policy_name,
                        episode_index,
                        first_seed + episode_index,
                        result.observations[0, 2],
                        result.is_success,
                        result.terminated,
                        result.truncated,
                        result.step_count,
                        result.total_reward,
                        final_observation[0],
                        final_observation[1],
                        abs(final_observation[3]),
                    )
                )


def save_metric_plot(summaries: dict[str, dict[str, float]]) -> None:
    """Plot the three most useful aggregate policy comparisons."""
    names = list(summaries)
    colors = ["tab:gray", "tab:blue"]
    metrics = (
        ("success_rate", "success rate", (0.0, 1.05)),
        ("mean_final_distance", "mean final distance [m]", None),
        ("mean_total_reward", "mean total reward", None),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for axis, (metric_key, title, limits) in zip(axes, metrics):
        values = [summaries[name][metric_key] for name in names]
        axis.bar(names, values, color=colors)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if limits is not None:
            axis.set_ylim(*limits)
    figure.suptitle("Random policy versus proportional-control baseline")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "policy_comparison.png", dpi=160)
    plt.close(figure)


def save_example_rollouts(
    policy_results: dict[str, list[EpisodeResult]],
    control_timestep: float,
) -> None:
    """Visualize position and action for the first shared target task."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
    for column, (policy_name, results) in enumerate(policy_results.items()):
        result = results[0]
        observation_times = np.arange(result.observations.shape[0]) * control_timestep
        action_times = np.arange(result.actions.shape[0]) * control_timestep
        target = float(result.observations[0, 2])

        axes[0, column].plot(observation_times, result.observations[:, 0])
        axes[0, column].axhline(
            target,
            color="tab:red",
            linestyle="--",
            label="target",
        )
        axes[0, column].set(title=policy_name, ylabel="position [m]")
        axes[0, column].legend()
        axes[1, column].plot(action_times, result.actions[:, 0])
        axes[1, column].set(xlabel="time [s]", ylabel="action")
        for axis in axes[:, column]:
            axis.grid(alpha=0.25)
    figure.suptitle("Example rollouts on the same target")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "example_rollouts.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Run the controlled comparison and save reproducible evidence."""
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=(
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "run.log", encoding="utf-8"),
        ),
    )

    policy_results = evaluate_policies(config)
    summaries = {
        name: asdict(summarize_episodes(results))
        for name, results in policy_results.items()
    }
    environment = make_environment(config)

    save_episode_table(policy_results, first_seed=int(config["seed"]))
    save_metric_plot(summaries)
    save_example_rollouts(policy_results, environment.control_timestep)

    output = {
        "experiment_name": config["experiment_name"],
        "seed": int(config["seed"]),
        "task_settings": {
            "control_timestep": environment.control_timestep,
            "max_episode_steps": environment.max_episode_steps,
            "target_distance_range": [
                environment.minimum_target_distance,
                environment.maximum_target_distance,
            ],
            "success_tolerance": environment.success_tolerance,
            "velocity_tolerance": environment.velocity_tolerance,
        },
        "policy_summaries": summaries,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    for name, summary in summaries.items():
        logging.info(
            "%s：成功率 %.1f%%，平均最终距离 %.4f m，平均回报 %.2f，平均步数 %.1f",
            name,
            100.0 * summary["success_rate"],
            summary["mean_final_distance"],
            summary["mean_total_reward"],
            summary["mean_step_count"],
        )
    logging.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
