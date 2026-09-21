"""Evaluate expert and BC recovery after a controlled action override."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_learning.behavior_cloning import (
    BehaviorCloningPolicy,
    load_behavior_cloning_checkpoint,
)
from robot_learning.distribution_shift import (
    run_episode_with_target_directed_override,
)
from robot_learning.gymnasium_rollout import (
    Policy,
    ProportionalReachPolicy,
    calculate_position_overshoot,
    summarize_episodes,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.trajectory_dataset import load_transition_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "028_distribution_shift.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "028_distribution_shift"


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def make_environment(
    config: dict[str, Any],
    environment_config: dict[str, Any],
) -> PointRobotReachEnv:
    """Recreate the same nominal task used for expert demonstrations."""
    return PointRobotReachEnv(
        PROJECT_ROOT / str(config["model_path"]),
        frame_skip=int(environment_config["frame_skip"]),
        max_episode_steps=int(environment_config["max_episode_steps"]),
        minimum_target_distance=float(
            environment_config["minimum_target_distance"]
        ),
        maximum_target_distance=float(
            environment_config["maximum_target_distance"]
        ),
        success_tolerance=float(environment_config["success_tolerance"]),
        velocity_tolerance=float(environment_config["velocity_tolerance"]),
        action_penalty_weight=float(
            environment_config["action_penalty_weight"]
        ),
    )


def main() -> None:
    """Run paired nominal and disturbed Episodes on unseen evaluation seeds."""
    config = load_json(CONFIG_PATH)
    source_config = load_json(
        PROJECT_ROOT / str(config["source_experiment_config_path"])
    )
    environment_config = load_json(
        PROJECT_ROOT / str(config["environment_config_path"])
    )
    dataset = load_transition_dataset(
        PROJECT_ROOT / str(config["dataset_path"])
    )
    manifest = load_json(
        PROJECT_ROOT / str(config["dataset_manifest_path"])
    )
    model, normalization, artifact = load_behavior_cloning_checkpoint(
        PROJECT_ROOT / str(config["checkpoint_path"]),
        device=str(config["device"]),
    )
    if artifact["source_dataset_sha256"] != manifest["dataset_content_sha256"]:
        raise ValueError("checkpoint and dataset manifest hashes do not match")

    episode_count = int(config["evaluation_episode_count"])
    first_seed = int(config["evaluation_seed_start"])
    if episode_count <= 0:
        raise ValueError("evaluation_episode_count must be positive")
    seeds = list(range(first_seed, first_seed + episode_count))
    if set(seeds) & set(dataset.environment_seeds.tolist()):
        raise ValueError("evaluation seeds overlap demonstration data")
    durations = [int(value) for value in config["override_duration_steps"]]
    if not durations or 0 not in durations or len(set(durations)) != len(durations):
        raise ValueError("override durations must be unique and include zero")

    environment = make_environment(config, environment_config)
    try:
        policies: dict[str, Policy] = {
            "proportional": ProportionalReachPolicy(
                float(environment_config["proportional_gain"]),
                environment.action_space,
            ),
            "behavior_cloning": BehaviorCloningPolicy(
                model,
                normalization,
                device=str(source_config["device"]),
            ),
        }
        all_results: dict[str, Any] = {}
        reference_targets: list[float] | None = None
        for policy_name, policy in policies.items():
            duration_results: dict[str, Any] = {}
            nominal_records: dict[int, dict[str, Any]] | None = None
            for duration in durations:
                disturbed = [
                    run_episode_with_target_directed_override(
                        environment,
                        policy,
                        seed=seed,
                        override_start_step=int(config["override_start_step"]),
                        override_duration_steps=duration,
                        override_magnitude=float(config["override_magnitude"]),
                    )
                    for seed in seeds
                ]
                episodes = [item.episode for item in disturbed]
                records = {
                    seed: {
                        "seed": seed,
                        "target_position": float(
                            item.episode.observations[0, 2]
                        ),
                        "success": item.episode.is_success,
                        "total_reward": item.episode.total_reward,
                        "step_count": item.episode.step_count,
                        "final_distance": abs(
                            float(item.episode.observations[-1, 3])
                        ),
                        "overshoot": calculate_position_overshoot(
                            item.episode.observations
                        ),
                        "override_applied_steps": item.override_applied_steps,
                    }
                    for seed, item in zip(seeds, disturbed)
                }
                targets = [records[seed]["target_position"] for seed in seeds]
                if reference_targets is None:
                    reference_targets = targets
                elif targets != reference_targets:
                    raise RuntimeError("conditions did not receive identical targets")
                if duration == 0:
                    nominal_records = records
                if nominal_records is None:
                    raise ValueError("zero-duration condition must be listed first")

                summary = asdict(summarize_episodes(episodes))
                summary["mean_overshoot"] = float(
                    np.mean([records[seed]["overshoot"] for seed in seeds])
                )
                summary["mean_step_delta_from_nominal"] = float(
                    np.mean(
                        [
                            records[seed]["step_count"]
                            - nominal_records[seed]["step_count"]
                            for seed in seeds
                        ]
                    )
                )
                summary["mean_reward_delta_from_nominal"] = float(
                    np.mean(
                        [
                            records[seed]["total_reward"]
                            - nominal_records[seed]["total_reward"]
                            for seed in seeds
                        ]
                    )
                )
                duration_results[str(duration)] = {
                    "summary": summary,
                    "episodes": [records[seed] for seed in seeds],
                }
            all_results[policy_name] = duration_results
    finally:
        environment.close()

    output = {
        "experiment_name": config["experiment_name"],
        "status": "controlled_disturbance_evaluation_complete",
        "evaluation_seeds": seeds,
        "training_seed_overlap_count": 0,
        "disturbance": {
            "type": "target_directed_action_override",
            "start_step": int(config["override_start_step"]),
            "durations": durations,
            "magnitude": float(config["override_magnitude"]),
        },
        "policy_results": all_results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    for policy_name, duration_results in all_results.items():
        print(policy_name)
        for duration in durations:
            summary = duration_results[str(duration)]["summary"]
            print(
                f"  duration={duration}: success={summary['success_rate']:.3f}, "
                f"steps={summary['mean_step_count']:.1f}, "
                f"step_delta={summary['mean_step_delta_from_nominal']:.1f}, "
                f"reward={summary['mean_total_reward']:.2f}"
            )


if __name__ == "__main__":
    main()
