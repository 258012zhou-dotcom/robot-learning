"""Check whether BC uses velocity even though the expert ignores it."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_learning.behavior_cloning import (
    BehaviorCloningPolicy,
    load_behavior_cloning_checkpoint,
)
from robot_learning.distribution_shift import evaluate_velocity_sweep
from robot_learning.gymnasium_rollout import ProportionalReachPolicy
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.trajectory_dataset import (
    TRAIN_SPLIT_ID,
    load_transition_dataset,
)


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


def main() -> None:
    """Sweep velocity while position, target, and target error stay fixed."""
    config = load_json(CONFIG_PATH)
    environment_config = load_json(
        PROJECT_ROOT / str(config["environment_config_path"])
    )
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
    model, normalization, _ = load_behavior_cloning_checkpoint(
        PROJECT_ROOT / str(config["checkpoint_path"]),
        device=str(config["device"]),
    )
    policies = {
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
    velocities = np.asarray(config["velocity_sweep"], dtype=np.float32)
    errors = np.asarray(config["target_error_sweep"], dtype=np.float32)
    dataset = load_transition_dataset(
        PROJECT_ROOT / str(config["dataset_path"])
    )
    train_expert_rows = (dataset.policy_ids == 1) & (
        dataset.split_ids == TRAIN_SPLIT_ID
    )
    train_velocities = dataset.observations[train_expert_rows, 1]
    train_velocity_range = [
        float(train_velocities.min()),
        float(train_velocities.max()),
    ]

    cases = []
    try:
        for error in errors:
            base_observation = np.asarray(
                [0.0, 0.0, error, error], dtype=np.float32
            )
            sweeps = {
                name: evaluate_velocity_sweep(
                    policy,
                    base_observation,
                    velocities,
                )
                for name, policy in policies.items()
            }
            expert_actions = sweeps["proportional"].actions[:, 0]
            bc_actions = sweeps["behavior_cloning"].actions[:, 0]
            inside_train_range = (
                (velocities >= train_velocity_range[0])
                & (velocities <= train_velocity_range[1])
            )
            cases.append(
                {
                    "target_error": float(error),
                    "velocities": velocities.tolist(),
                    "proportional_actions": expert_actions.tolist(),
                    "behavior_cloning_actions": bc_actions.tolist(),
                    "expert_action_span": float(np.ptp(expert_actions)),
                    "bc_action_span_full_sweep": float(np.ptp(bc_actions)),
                    "bc_action_span_inside_train_velocity_range": float(
                        np.ptp(bc_actions[inside_train_range])
                    ),
                    "maximum_absolute_bc_expert_difference": float(
                        np.max(np.abs(bc_actions - expert_actions))
                    ),
                }
            )
    finally:
        environment.close()

    output = {
        "experiment_name": config["experiment_name"],
        "analysis": "velocity_sensitivity_with_fixed_error",
        "train_expert_velocity_range": train_velocity_range,
        "cases": cases,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "velocity_sensitivity.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    for case in cases:
        print(
            f"error={case['target_error']:+.2f}, "
            f"expert_span={case['expert_action_span']:.6f}, "
            f"bc_train_range_span="
            f"{case['bc_action_span_inside_train_velocity_range']:.6f}, "
            f"bc_full_span={case['bc_action_span_full_sweep']:.6f}, "
            f"max_difference="
            f"{case['maximum_absolute_bc_expert_difference']:.6f}"
        )


if __name__ == "__main__":
    main()
