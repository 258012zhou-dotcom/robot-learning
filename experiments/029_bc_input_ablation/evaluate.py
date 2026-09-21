"""Compare full-observation and target-error-only BC policies."""

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
    evaluate_velocity_sweep,
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
CONFIG_PATH = PROJECT_ROOT / "configs" / "029_bc_input_ablation.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "029_bc_input_ablation"


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    """Run paired nominal, disturbed, and velocity-sensitivity evaluations."""
    config = load_json(CONFIG_PATH)
    environment_config = load_json(
        PROJECT_ROOT / str(config["environment_config_path"])
    )
    manifest = load_json(
        PROJECT_ROOT / str(config["dataset_manifest_path"])
    )
    dataset = load_transition_dataset(
        PROJECT_ROOT / str(config["dataset_path"])
    )
    model_specs = {
        "full_observation_bc": PROJECT_ROOT
        / str(config["full_observation_checkpoint_path"]),
        "target_error_only_bc": OUTPUT_DIR / "best_model.pt",
    }
    loaded_models = {
        name: load_behavior_cloning_checkpoint(path, device=str(config["device"]))
        for name, path in model_specs.items()
    }
    for _, _, artifact in loaded_models.values():
        if artifact["source_dataset_sha256"] != manifest["dataset_content_sha256"]:
            raise ValueError("checkpoint and dataset manifest hashes do not match")

    environment = PointRobotReachEnv(
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
    policies: dict[str, Policy] = {
        "proportional": ProportionalReachPolicy(
            float(environment_config["proportional_gain"]),
            environment.action_space,
        )
    }
    for name, (model, normalization, artifact) in loaded_models.items():
        policies[name] = BehaviorCloningPolicy(
            model,
            normalization,
            observation_indices=artifact.get("observation_indices"),
            device=str(config["device"]),
        )

    first_seed = int(config["evaluation_seed_start"])
    episode_count = int(config["evaluation_episode_count"])
    seeds = list(range(first_seed, first_seed + episode_count))
    if set(seeds) & set(dataset.environment_seeds.tolist()):
        raise ValueError("evaluation seeds overlap demonstration data")
    durations = [int(value) for value in config["override_duration_steps"]]
    if not durations or durations[0] != 0:
        raise ValueError("override durations must start with zero")

    policy_results: dict[str, Any] = {}
    try:
        for policy_name, policy in policies.items():
            duration_results = {}
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
                summary = asdict(summarize_episodes(episodes))
                summary["mean_overshoot"] = float(
                    np.mean(
                        [
                            calculate_position_overshoot(item.observations)
                            for item in episodes
                        ]
                    )
                )
                duration_results[str(duration)] = summary

            velocity_cases = []
            velocities = np.asarray(config["velocity_sweep"], dtype=np.float32)
            for error in config["target_error_sweep"]:
                sweep = evaluate_velocity_sweep(
                    policy,
                    np.asarray([0.0, 0.0, error, error], dtype=np.float32),
                    velocities,
                )
                velocity_cases.append(
                    {
                        "target_error": float(error),
                        "action_span": float(np.ptp(sweep.actions[:, 0])),
                    }
                )
            policy_results[policy_name] = {
                "disturbance_results": duration_results,
                "velocity_sensitivity": velocity_cases,
            }
    finally:
        environment.close()

    full_offline = load_json(
        PROJECT_ROOT / str(config["full_observation_results_path"])
    )
    target_only_offline = load_json(OUTPUT_DIR / "offline_results.json")
    output = {
        "experiment_name": config["experiment_name"],
        "status": "input_ablation_complete",
        "controlled_variable": "observation_features",
        "shared_training_seed": int(config["seed"]),
        "evaluation_seeds": seeds,
        "offline_test_mse": {
            "full_observation_bc": float(full_offline["test_mse"]),
            "target_error_only_bc": float(target_only_offline["test_mse"]),
        },
        "policy_results": policy_results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "comparison_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("offline test MSE")
    for name, mse in output["offline_test_mse"].items():
        print(f"  {name}: {mse:.8g}")
    for name, result in policy_results.items():
        spans = [case["action_span"] for case in result["velocity_sensitivity"]]
        print(f"{name}: maximum_velocity_action_span={max(spans):.6f}")
        for duration in durations:
            summary = result["disturbance_results"][str(duration)]
            print(
                f"  duration={duration}: "
                f"success={summary['success_rate']:.3f}, "
                f"steps={summary['mean_step_count']:.1f}"
            )


if __name__ == "__main__":
    main()
