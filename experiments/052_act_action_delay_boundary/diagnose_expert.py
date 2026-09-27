"""Post-hoc diagnostic: test the original P expert under the same delay grid."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from robot_learning.gymnasium_rollout import (
    ProportionalReachPolicy,
    run_episode,
    summarize_episodes,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "052_act_action_delay_boundary.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "052_act_action_delay_boundary"


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    with (PROJECT_ROOT / str(config["environment_config_path"])).open(
        encoding="utf-8"
    ) as file:
        environment_config = json.load(file)
    with (OUTPUT_DIR / "results.json").open(encoding="utf-8") as file:
        act_results = json.load(file)
    seeds = act_results["evaluation_seeds"]
    environment = PointRobotReachEnv(
        PROJECT_ROOT / str(config["model_path"]),
        frame_skip=int(environment_config["frame_skip"]),
        max_episode_steps=int(environment_config["max_episode_steps"]),
        minimum_target_distance=float(environment_config["minimum_target_distance"]),
        maximum_target_distance=float(environment_config["maximum_target_distance"]),
        success_tolerance=float(environment_config["success_tolerance"]),
        velocity_tolerance=float(environment_config["velocity_tolerance"]),
        action_penalty_weight=float(environment_config["action_penalty_weight"]),
    )
    policy = ProportionalReachPolicy(
        float(environment_config["proportional_gain"]), environment.action_space
    )
    conditions = {}
    try:
        for delay in act_results["action_delay_steps"]:
            episodes = []
            records = []
            for seed in seeds:
                result = run_episode(
                    environment,
                    policy,
                    seed=seed,
                    reset_options={"action_delay_steps": delay},
                )
                episodes.append(result)
                records.append(
                    {
                        "seed": seed,
                        "target_position": float(result.observations[0, 2]),
                        "success": result.is_success,
                        "step_count": result.step_count,
                        "final_distance": abs(float(result.observations[-1, 3])),
                        "final_velocity": float(result.observations[-1, 1]),
                    }
                )
            act_targets = [
                row["target_position"]
                for row in act_results["conditions"][str(delay)]["latest_first_action"]["episodes"]
            ]
            if [row["target_position"] for row in records] != act_targets:
                raise ValueError("expert and ACT received different targets")
            conditions[str(delay)] = {
                "summary": asdict(summarize_episodes(episodes)),
                "mean_abs_final_velocity": float(
                    np.mean([abs(row["final_velocity"]) for row in records])
                ),
                "episodes": records,
            }
    finally:
        environment.close()
    output = {
        "diagnostic": "post_hoc_original_p_expert_same_delay_grid",
        "evaluation_seeds": seeds,
        "action_delay_steps": act_results["action_delay_steps"],
        "conditions": conditions,
    }
    with (OUTPUT_DIR / "expert_diagnostic.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    for delay, item in conditions.items():
        summary = item["summary"]
        print(
            f"P expert delay={delay}: "
            f"success={summary['success_count']}/{summary['episode_count']} "
            f"distance={summary['mean_final_distance']:.4f} "
            f"steps={summary['mean_step_count']:.1f}"
        )


if __name__ == "__main__":
    main()
