"""Measure how control frequency changes one fixed P-control policy."""

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
    calculate_position_overshoot,
    run_episode,
    summarize_episodes,
)
from robot_learning.mujoco_basics import load_mujoco_model
from robot_learning.point_robot_reach_env import PointRobotReachEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "019_control_frequency.json"
MODEL_PATH = (
    PROJECT_ROOT / "experiments" / "017_mujoco_step" / "point_robot.xml"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "019_control_frequency"


def load_config() -> dict[str, Any]:
    """Read settings and reject empty or invalid comparisons."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    frame_skips = [int(value) for value in config["frame_skips"]]
    if not frame_skips or any(value <= 0 for value in frame_skips):
        raise ValueError("frame_skips must contain positive integers")
    if int(config["episode_count"]) <= 0:
        raise ValueError("episode_count must be positive")
    if float(config["maximum_duration_seconds"]) <= 0.0:
        raise ValueError("maximum_duration_seconds must be positive")
    return config


def make_environment(
    config: dict[str, Any],
    *,
    frame_skip: int,
    physics_timestep: float,
) -> PointRobotReachEnv:
    """Keep maximum simulated duration equal across control frequencies."""
    control_timestep = physics_timestep * frame_skip
    maximum_duration = float(config["maximum_duration_seconds"])
    max_episode_steps = round(maximum_duration / control_timestep)
    if not np.isclose(max_episode_steps * control_timestep, maximum_duration):
        raise ValueError("maximum duration must be divisible by control timestep")

    return PointRobotReachEnv(
        MODEL_PATH,
        frame_skip=frame_skip,
        max_episode_steps=max_episode_steps,
        minimum_target_distance=float(config["minimum_target_distance"]),
        maximum_target_distance=float(config["maximum_target_distance"]),
        success_tolerance=float(config["success_tolerance"]),
        velocity_tolerance=float(config["velocity_tolerance"]),
        action_penalty_weight=float(config["action_penalty_weight"]),
    )


def evaluate_frequencies(
    config: dict[str, Any],
) -> tuple[
    dict[str, list[EpisodeResult]],
    dict[str, dict[str, int | float]],
]:
    """Evaluate one unchanged policy on identical targets at each frequency."""
    model = load_mujoco_model(MODEL_PATH)
    physics_timestep = float(model.opt.timestep)
    first_seed = int(config["seed"])
    episode_seeds = range(
        first_seed,
        first_seed + int(config["episode_count"]),
    )

    all_results: dict[str, list[EpisodeResult]] = {}
    summaries: dict[str, dict[str, int | float]] = {}
    for frame_skip in [int(value) for value in config["frame_skips"]]:
        environment = make_environment(
            config,
            frame_skip=frame_skip,
            physics_timestep=physics_timestep,
        )
        policy = ProportionalReachPolicy(
            float(config["proportional_gain"]),
            environment.action_space,
        )
        results = [
            run_episode(environment, policy, seed=episode_seed)
            for episode_seed in episode_seeds
        ]

        label = f"{environment.control_timestep:g}_seconds"
        base_summary = asdict(summarize_episodes(results))
        summaries[label] = {
            "frame_skip": frame_skip,
            "control_timestep": environment.control_timestep,
            "control_frequency_hz": 1.0 / environment.control_timestep,
            "max_episode_steps": environment.max_episode_steps,
            **base_summary,
            "mean_episode_duration_seconds": float(
                np.mean(
                    [
                        result.step_count * environment.control_timestep
                        for result in results
                    ]
                )
            ),
            # Raw return depends on how many rewards are emitted per second.
            # Multiplying by dt approximates a comparable continuous-time cost.
            "mean_time_integrated_reward": float(
                np.mean(
                    [
                        result.total_reward * environment.control_timestep
                        for result in results
                    ]
                )
            ),
            "mean_position_overshoot": float(
                np.mean(
                    [
                        calculate_position_overshoot(result.observations)
                        for result in results
                    ]
                )
            ),
        }
        all_results[label] = results

    # Each frequency must receive the same ordered set of target positions.
    reference_targets = next(iter(all_results.values()))
    expected = np.asarray([result.observations[0, 2] for result in reference_targets])
    for results in all_results.values():
        actual = np.asarray([result.observations[0, 2] for result in results])
        if not np.array_equal(actual, expected):
            raise RuntimeError("frequency evaluations used different targets")
    return all_results, summaries


def save_episode_table(
    all_results: dict[str, list[EpisodeResult]],
    summaries: dict[str, dict[str, int | float]],
    *,
    first_seed: int,
) -> None:
    """Save one row per frequency and Episode for later error analysis."""
    with (OUTPUT_DIR / "episodes.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "control_frequency_hz",
                "episode_index",
                "environment_seed",
                "target_position",
                "success",
                "step_count",
                "duration_seconds",
                "total_reward",
                "time_integrated_reward",
                "final_distance",
                "position_overshoot",
            )
        )
        for label, results in all_results.items():
            control_timestep = float(summaries[label]["control_timestep"])
            frequency = float(summaries[label]["control_frequency_hz"])
            for episode_index, result in enumerate(results):
                writer.writerow(
                    (
                        frequency,
                        episode_index,
                        first_seed + episode_index,
                        result.observations[0, 2],
                        result.is_success,
                        result.step_count,
                        result.step_count * control_timestep,
                        result.total_reward,
                        result.total_reward * control_timestep,
                        abs(result.observations[-1, 3]),
                        calculate_position_overshoot(result.observations),
                    )
                )


def save_metric_plot(summaries: dict[str, dict[str, int | float]]) -> None:
    """Compare task quality and computational decision count together."""
    labels = [f'{value["control_frequency_hz"]:g} Hz' for value in summaries.values()]
    metrics = (
        ("success_rate", "success rate"),
        ("mean_episode_duration_seconds", "mean completion time [s]"),
        ("mean_position_overshoot", "mean overshoot [m]"),
        ("mean_step_count", "mean policy decisions"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (metric_key, title) in zip(axes.flat, metrics):
        values = [float(summary[metric_key]) for summary in summaries.values()]
        axis.plot(labels, values, marker="o")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylim(0.0, 1.05)
    figure.suptitle("Effect of control frequency with fixed P gain")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "frequency_comparison.png", dpi=160)
    plt.close(figure)


def save_example_trajectories(
    all_results: dict[str, list[EpisodeResult]],
    summaries: dict[str, dict[str, int | float]],
) -> None:
    """Overlay the first shared target trajectory at every frequency."""
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    target = float(next(iter(all_results.values()))[0].observations[0, 2])
    for label, results in all_results.items():
        result = results[0]
        control_timestep = float(summaries[label]["control_timestep"])
        frequency = float(summaries[label]["control_frequency_hz"])
        observation_times = np.arange(result.observations.shape[0]) * control_timestep
        action_times = np.arange(result.actions.shape[0]) * control_timestep
        axes[0].plot(
            observation_times,
            result.observations[:, 0],
            label=f"{frequency:g} Hz",
        )
        axes[1].step(
            action_times,
            result.actions[:, 0],
            where="post",
            label=f"{frequency:g} Hz",
        )
    axes[0].axhline(target, color="black", linestyle="--", label="target")
    axes[0].set_ylabel("position [m]")
    axes[1].set(xlabel="simulation time [s]", ylabel="action")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Same P controller and target at different control rates")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "example_trajectories.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Run the controlled frequency comparison and save its evidence."""
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

    all_results, summaries = evaluate_frequencies(config)
    save_episode_table(
        all_results,
        summaries,
        first_seed=int(config["seed"]),
    )
    save_metric_plot(summaries)
    save_example_trajectories(all_results, summaries)

    output = {
        "experiment_name": config["experiment_name"],
        "seed": int(config["seed"]),
        "episode_count_per_frequency": int(config["episode_count"]),
        "maximum_duration_seconds": float(config["maximum_duration_seconds"]),
        "proportional_gain": float(config["proportional_gain"]),
        "frequency_summaries": summaries,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    for summary in summaries.values():
        logging.info(
            "%.0f Hz：成功率 %.1f%%，平均完成时间 %.3f s，平均超调 %.4f m，平均决策 %.1f 次",
            summary["control_frequency_hz"],
            100.0 * summary["success_rate"],
            summary["mean_episode_duration_seconds"],
            summary["mean_position_overshoot"],
            summary["mean_step_count"],
        )
    logging.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
