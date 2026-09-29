"""Freeze the route-conditioned policy and test a double-dt interface error."""

import hashlib
import json
from pathlib import Path

import numpy as np

from robot_learning.gymnasium_rollout import run_episode
from robot_learning.route_conditioning import (
    RouteConditionedSequenceBCPolicy,
    route_at_obstacle_x,
)
from robot_learning.sequence_behavior_cloning import load_sequence_checkpoint
from robot_learning.two_route_navigation import TwoRouteNavigationEnv, sample_scene
from robot_learning.velocity_command_adapter import VelocityCommandAdapter


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/055_velocity_unit_boundary.json"
OUTPUT_DIR = ROOT / "outputs/055_velocity_unit_boundary"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def scene_seeds(task: dict, split: str) -> list[int]:
    source = task["scene_splits"][split]
    start = int(source["seed_start"])
    return list(range(start, start + int(source["scene_count"])))


def evaluate(
    environment: TwoRouteNavigationEnv,
    adapter: VelocityCommandAdapter,
    seeds: list[int],
) -> dict:
    records = []
    for scene_seed in seeds:
        for route in (-1, 1):
            adapter.reset(route=route)
            result = run_episode(environment, adapter, seed=scene_seed)
            raw = np.stack(adapter.raw_commands)
            if not np.allclose(result.actions, raw * adapter.scale, atol=1e-7):
                raise RuntimeError("adapter output is not raw velocity times scale")
            observed_displacement = np.diff(result.observations[:, :2], axis=0)
            if not np.allclose(
                observed_displacement, result.actions * environment.dt, atol=1e-6
            ):
                raise RuntimeError("environment displacement violates velocity contract")
            actual_route = route_at_obstacle_x(result.observations)
            records.append({
                "scene_seed": scene_seed,
                "requested_route": route,
                "actual_route": actual_route,
                "task_success": bool(result.is_success),
                "route_match": actual_route == route,
                "joint_success": bool(result.is_success and actual_route == route),
                "collision": bool(result.terminated and not result.is_success),
                "timeout": bool(result.truncated),
                "steps": result.step_count,
                "first_policy_velocity": raw[0].tolist(),
                "first_applied_velocity": result.actions[0].tolist(),
                "first_actual_displacement": observed_displacement[0].tolist(),
                "final_distance": float(np.linalg.norm(
                    result.observations[-1, :2] - result.observations[0, 2:4]
                )),
            })
    return {
        "scene_count": len(seeds),
        "request_count": len(records),
        "task_success_count": sum(row["task_success"] for row in records),
        "joint_success_count": sum(row["joint_success"] for row in records),
        "collision_count": sum(row["collision"] for row in records),
        "timeout_count": sum(row["timeout"] for row in records),
        "mean_first_displacement_x": float(np.mean([
            row["first_actual_displacement"][0] for row in records
        ])),
        "requests": records,
    }


def main() -> None:
    config = read_json(CONFIG_PATH)
    task = read_json(ROOT / config["source_task_config_path"])
    bc_config = read_json(ROOT / config["source_bc_config_path"])
    source_manifest = read_json(ROOT / config["source_manifest_path"])
    source_report = read_json(ROOT / config["source_result_path"])
    dataset_path = ROOT / config["source_dataset_path"]
    source_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if (source_hash != source_manifest["dataset_sha256"] or
            source_hash != source_report["source_dataset_sha256"]):
        raise RuntimeError("source dataset identity changed")
    model, normalization, artifact = load_sequence_checkpoint(
        ROOT / config["source_checkpoint_path"]
    )
    if (artifact["source_dataset_sha256"] != source_hash or
            artifact["seed"] != bc_config["seed"] or
            model.observation_size != 8 or
            model.horizon != bc_config["prediction_horizon"]):
        raise RuntimeError("frozen checkpoint does not match experiment 054")
    environment = TwoRouteNavigationEnv(
        dt=float(task["dt"]),
        max_episode_steps=int(task["max_episode_steps"]),
        goal_tolerance=float(task["goal_tolerance"]),
    )
    conditions = {"correct_velocity": 1.0, "double_applied_dt": environment.dt}
    comparisons = {}
    try:
        for split in ("validation", "test"):
            comparisons[split] = {}
            for name, scale in conditions.items():
                policy = RouteConditionedSequenceBCPolicy(
                    model, normalization,
                    execution_horizon=int(bc_config["execution_horizon"]),
                )
                adapter = VelocityCommandAdapter(policy, scale=scale)
                comparisons[split][name] = evaluate(
                    environment, adapter, scene_seeds(task, split)
                )
            correct = comparisons[split]["correct_velocity"]["requests"]
            incorrect = comparisons[split]["double_applied_dt"]["requests"]
            for normal, wrong in zip(correct, incorrect, strict=True):
                if (normal["scene_seed"] != wrong["scene_seed"] or
                        normal["requested_route"] != wrong["requested_route"] or
                        not np.allclose(normal["first_policy_velocity"],
                                        wrong["first_policy_velocity"], atol=1e-7)):
                    raise RuntimeError("conditions are not paired at the first observation")
    finally:
        environment.close()
    correct_test = comparisons["test"]["correct_velocity"]["requests"]
    reference_test = source_report["comparisons"]["test"]["conditioned_bc"]["requests"]
    for new, reference in zip(correct_test, reference_test, strict=True):
        if (new["scene_seed"] != reference["scene_seed"] or
                new["requested_route"] != reference["requested_route"] or
                new["task_success"] != reference["task_success"] or
                new["joint_success"] != reference["joint_success"] or
                new["steps"] != reference["step_count"] or
                not np.allclose(new["first_applied_velocity"],
                                reference["first_action"], atol=1e-7)):
            raise RuntimeError("correct velocity condition did not reproduce experiment 054")
    test_scenes = [sample_scene(seed) for seed in scene_seeds(task, "test")]
    minimum_required_x_displacement = min(
        scene.goal_x - environment.goal_tolerance - scene.start_x
        for scene in test_scenes
    )
    maximum_wrong_x_displacement = (
        environment.max_episode_steps
        * environment.dt
        * conditions["double_applied_dt"]
        * float(environment.action_space.high[0])
    )
    report = {
        "status": "frozen_policy_synthetic_velocity_unit_mismatch",
        "source_dataset_sha256": source_hash,
        "source_checkpoint": config["source_checkpoint_path"],
        "dt_seconds": environment.dt,
        "max_episode_steps": environment.max_episode_steps,
        "conditions": conditions,
        "reachability_bound": {
            "minimum_required_x_displacement": minimum_required_x_displacement,
            "maximum_wrong_x_displacement": maximum_wrong_x_displacement,
            "wrong_condition_cannot_reach_goal_by_x": (
                maximum_wrong_x_displacement < minimum_required_x_displacement
            ),
        },
        "comparisons": comparisons,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        name: {key: value for key, value in result.items() if key != "requests"}
        for name, result in comparisons["test"].items()
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
