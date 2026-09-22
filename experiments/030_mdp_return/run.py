"""Show how discounting changes returns in a tiny MDP and one rollout."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_learning.gymnasium_rollout import ProportionalReachPolicy, run_episode
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.returns import calculate_discounted_returns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "030_mdp_return.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "030_mdp_return"


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def compare_tiny_mdp(
    quick_rewards: np.ndarray,
    patient_rewards: np.ndarray,
    discount_factors: list[float],
) -> list[dict[str, float | str]]:
    """Compare an immediate reward with a larger one-step-delayed reward."""
    rows = []
    for gamma in discount_factors:
        quick_return = float(
            calculate_discounted_returns(quick_rewards, gamma)[0]
        )
        patient_return = float(
            calculate_discounted_returns(patient_rewards, gamma)[0]
        )
        if np.isclose(quick_return, patient_return):
            preferred_choice = "tie"
        elif quick_return > patient_return:
            preferred_choice = "quick"
        else:
            preferred_choice = "patient"
        rows.append(
            {
                "discount_factor": gamma,
                "quick_return": quick_return,
                "patient_return": patient_return,
                "preferred_choice": preferred_choice,
            }
        )
    return rows


def make_environment(
    config: dict[str, Any],
    environment_config: dict[str, Any],
) -> PointRobotReachEnv:
    """Recreate the point-robot MDP used by the preceding experiments."""
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
    """Evaluate transparent reward sequences before any RL optimization."""
    config = load_json(CONFIG_PATH)
    environment_config = load_json(
        PROJECT_ROOT / str(config["environment_config_path"])
    )
    discount_factors = [float(value) for value in config["discount_factors"]]
    tiny_config = config["tiny_mdp"]
    tiny_results = compare_tiny_mdp(
        np.asarray(tiny_config["quick_rewards"], dtype=np.float64),
        np.asarray(tiny_config["patient_rewards"], dtype=np.float64),
        discount_factors,
    )

    robot_config = config["point_robot"]
    environment = make_environment(config, environment_config)
    policy = ProportionalReachPolicy(
        float(robot_config["proportional_gain"]),
        environment.action_space,
    )
    try:
        episode = run_episode(
            environment,
            policy,
            seed=int(robot_config["seed"]),
            reset_options={
                "initial_position": float(robot_config["initial_position"]),
                "target_position": float(robot_config["target_position"]),
            },
        )
    finally:
        environment.close()

    robot_returns = []
    for gamma in discount_factors:
        returns = calculate_discounted_returns(episode.rewards, gamma)
        residuals = returns[:-1] - (
            episode.rewards[:-1] + gamma * returns[1:]
        )
        robot_returns.append(
            {
                "discount_factor": gamma,
                "initial_return": float(returns[0]),
                "final_return": float(returns[-1]),
                "maximum_recursion_residual": float(
                    np.max(np.abs(residuals)) if residuals.size else 0.0
                ),
            }
        )

    output = {
        "experiment_name": config["experiment_name"],
        "status": "mdp_return_demonstration_complete",
        "mdp_components": {
            "state_or_observation": (
                "position, velocity, target_position, target_error"
            ),
            "action": "one bounded motor command",
            "transition": "MuJoCo physics over one control step",
            "reward": "negative distance minus action penalty",
            "discount_factor": "configured gamma",
            "policy": "proportional controller for this demonstration",
        },
        "tiny_mdp": {
            "quick_rewards": tiny_config["quick_rewards"],
            "patient_rewards": tiny_config["patient_rewards"],
            "results": tiny_results,
        },
        "point_robot_episode": {
            "seed": int(robot_config["seed"]),
            "target_position": float(robot_config["target_position"]),
            "success": episode.is_success,
            "step_count": episode.step_count,
            "undiscounted_total_reward": episode.total_reward,
            "returns": robot_returns,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("tiny MDP: quick=[1], patient=[0, 2]")
    for row in tiny_results:
        print(
            f"  gamma={row['discount_factor']:.2f}: "
            f"quick={row['quick_return']:.3f}, "
            f"patient={row['patient_return']:.3f}, "
            f"preferred={row['preferred_choice']}"
        )
    print(
        f"point robot: success={episode.is_success}, "
        f"steps={episode.step_count}, total_reward={episode.total_reward:.3f}"
    )
    for row in robot_returns:
        print(
            f"  gamma={row['discount_factor']:.2f}: "
            f"G0={row['initial_return']:.3f}"
        )


if __name__ == "__main__":
    main()
