"""Compare Random, P-control, and loaded BC policies in closed loop."""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_learning.behavior_cloning import (
    BehaviorCloningPolicy,
    load_behavior_cloning_checkpoint,
)
from robot_learning.gymnasium_rollout import (
    Policy,
    ProportionalReachPolicy,
    RandomPolicy,
    calculate_position_overshoot,
    run_episode,
    summarize_episodes,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.trajectory_dataset import load_transition_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "027_behavior_cloning.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "027_behavior_cloning"


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
    """Recreate the environment used to collect the expert dataset."""
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


def evaluate_policy(
    environment: PointRobotReachEnv,
    policy: Policy,
    seeds: list[int],
) -> tuple[list[dict[str, int | float | bool]], dict[str, int | float]]:
    """Keep per-Episode evidence and calculate aggregate task metrics."""
    results = [
        run_episode(environment, policy, seed=seed)
        for seed in seeds
    ]
    episodes = [
        {
            "seed": seed,
            "target_position": float(result.observations[0, 2]),
            "target_direction": (
                1 if result.observations[0, 2] > 0.0 else -1
            ),
            "success": result.is_success,
            "total_reward": result.total_reward,
            "step_count": result.step_count,
            "final_distance": abs(float(result.observations[-1, 3])),
            "overshoot": calculate_position_overshoot(result.observations),
        }
        for seed, result in zip(seeds, results)
    ]
    summary = asdict(summarize_episodes(results))
    summary["mean_overshoot"] = float(
        np.mean([episode["overshoot"] for episode in episodes])
    )
    return episodes, summary


def main() -> None:
    """Load the saved policy and compare all baselines on unseen seeds."""
    config = load_json(CONFIG_PATH)
    environment_config = load_json(
        PROJECT_ROOT / str(config["environment_config_path"])
    )
    dataset = load_transition_dataset(
        PROJECT_ROOT / str(config["dataset_path"])
    )
    first_seed = int(config["evaluation_seed_start"])
    episode_count = int(config["evaluation_episode_count"])
    if episode_count <= 0:
        raise ValueError("evaluation_episode_count must be positive")
    seeds = list(range(first_seed, first_seed + episode_count))
    if set(seeds) & set(dataset.environment_seeds.tolist()):
        raise ValueError("evaluation seeds overlap the demonstration dataset")

    model, normalization, artifact = load_behavior_cloning_checkpoint(
        OUTPUT_DIR / "best_model.pt",
        device=str(config["device"]),
    )
    manifest = load_json(
        PROJECT_ROOT / str(config["dataset_manifest_path"])
    )
    if artifact["source_dataset_sha256"] != manifest["dataset_content_sha256"]:
        raise ValueError("checkpoint and dataset manifest hashes do not match")

    environment = make_environment(config, environment_config)
    try:
        policies: dict[str, Policy] = {
            "random": RandomPolicy(
                environment.action_space,
                seed=int(config["random_policy_seed"]),
            ),
            "proportional": ProportionalReachPolicy(
                float(environment_config["proportional_gain"]),
                environment.action_space,
            ),
            "behavior_cloning": BehaviorCloningPolicy(
                model,
                normalization,
                device=str(config["device"]),
            ),
        }
        policy_results = {}
        reference_targets: list[float] | None = None
        for name, policy in policies.items():
            episodes, summary = evaluate_policy(environment, policy, seeds)
            targets = [float(episode["target_position"]) for episode in episodes]
            if reference_targets is None:
                reference_targets = targets
            elif targets != reference_targets:
                raise RuntimeError("policies did not receive identical targets")
            policy_results[name] = {
                "summary": summary,
                "episodes": episodes,
            }
    finally:
        environment.close()

    assert reference_targets is not None
    evaluation = {
        "experiment_name": config["experiment_name"],
        "status": "closed_loop_evaluation_complete",
        "checkpoint": "best_model.pt",
        "evaluation_seeds": seeds,
        "training_seed_overlap_count": 0,
        "target_direction_counts": {
            "negative": sum(target < 0.0 for target in reference_targets),
            "positive": sum(target > 0.0 for target in reference_targets),
        },
        "policy_results": policy_results,
    }
    with (OUTPUT_DIR / "closed_loop_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(evaluation, file, ensure_ascii=False, indent=2)

    for name, result in policy_results.items():
        summary = result["summary"]
        print(
            f"{name}: success_rate={summary['success_rate']:.3f}, "
            f"mean_final_distance={summary['mean_final_distance']:.6f}, "
            f"mean_reward={summary['mean_total_reward']:.3f}, "
            f"mean_steps={summary['mean_step_count']:.1f}"
        )


if __name__ == "__main__":
    main()
