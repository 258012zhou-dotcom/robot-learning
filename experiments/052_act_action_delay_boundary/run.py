"""Bounded action-delay stress check for the frozen low-dimensional ACT policy."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_learning.act_lowdim_policy import (
    ACTLowDimPolicy,
    load_act_lowdim_checkpoint,
)
from robot_learning.gymnasium_rollout import (
    calculate_position_overshoot,
    run_episode,
    summarize_episodes,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.trajectory_dataset import load_transition_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "052_act_action_delay_boundary.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "052_act_action_delay_boundary"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def make_environment(
    config: dict[str, Any], environment_config: dict[str, Any]
) -> PointRobotReachEnv:
    return PointRobotReachEnv(
        PROJECT_ROOT / str(config["model_path"]),
        frame_skip=int(environment_config["frame_skip"]),
        max_episode_steps=int(environment_config["max_episode_steps"]),
        minimum_target_distance=float(environment_config["minimum_target_distance"]),
        maximum_target_distance=float(environment_config["maximum_target_distance"]),
        success_tolerance=float(environment_config["success_tolerance"]),
        velocity_tolerance=float(environment_config["velocity_tolerance"]),
        action_penalty_weight=float(environment_config["action_penalty_weight"]),
    )


def evaluate(
    environment: PointRobotReachEnv,
    policy: ACTLowDimPolicy,
    seeds: list[int],
    *,
    delay_steps: int,
    failure_traces: dict[str, np.ndarray],
    policy_name: str,
) -> dict[str, Any]:
    episodes = []
    records = []
    for seed in seeds:
        policy.reset()
        result = run_episode(
            environment,
            policy,
            seed=seed,
            reset_options={"action_delay_steps": delay_steps},
        )
        episodes.append(result)
        final_observation = result.observations[-1]
        records.append(
            {
                "seed": seed,
                "target_position": float(result.observations[0, 2]),
                "success": result.is_success,
                "step_count": result.step_count,
                "total_reward": result.total_reward,
                "final_distance": abs(float(final_observation[3])),
                "final_velocity": float(final_observation[1]),
                "overshoot": calculate_position_overshoot(result.observations),
            }
        )
        if not result.is_success:
            prefix = f"{policy_name}_delay_{delay_steps}_seed_{seed}"
            failure_traces[f"{prefix}_observations"] = result.observations
            failure_traces[f"{prefix}_actions"] = result.actions
    summary = asdict(summarize_episodes(episodes))
    summary["mean_overshoot"] = float(np.mean([r["overshoot"] for r in records]))
    return {"summary": summary, "episodes": records}


def verify_nominal_matches_reference(
    current: dict[str, Any], previous: dict[str, Any]
) -> None:
    for new, old in zip(current["episodes"], previous["episodes"], strict=True):
        if (new["seed"], new["success"], new["step_count"]) != (
            old["seed"], old["success"], old["step_count"]
        ):
            raise ValueError("nominal ACT result differs from experiment 051")
        for key in ("target_position", "total_reward", "final_distance"):
            if not np.isclose(new[key], old[key], rtol=0.0, atol=1e-6):
                raise ValueError(f"nominal ACT {key} differs from experiment 051")


def main() -> None:
    config = read_json(CONFIG_PATH)
    if config["device"] != "cpu":
        raise ValueError("this teaching evaluation uses CPU only")
    torch.set_num_threads(1)
    checkpoint_path = PROJECT_ROOT / str(config["act_checkpoint_path"])
    model, normalization, artifact = load_act_lowdim_checkpoint(checkpoint_path)
    manifest = read_json(PROJECT_ROOT / str(config["dataset_manifest_path"]))
    if artifact["source_dataset_sha256"] != manifest["dataset_content_sha256"]:
        raise ValueError("ACT checkpoint and source dataset identity differ")
    dataset = load_transition_dataset(PROJECT_ROOT / str(config["dataset_path"]))
    first_seed = int(config["evaluation_seed_start"])
    episode_count = int(config["evaluation_episode_count"])
    if episode_count <= 0:
        raise ValueError("evaluation_episode_count must be positive")
    seeds = list(range(first_seed, first_seed + episode_count))
    if set(seeds) & set(dataset.environment_seeds.tolist()):
        raise ValueError("evaluation seeds overlap demonstration data")
    delays = [int(value) for value in config["action_delay_steps"]]
    if delays != [0, 3, 10]:
        raise ValueError("this preregistered teaching check uses delays 0, 3, 10")
    reference = read_json(PROJECT_ROOT / str(config["reference_closed_loop_path"]))
    if reference["evaluation_seeds"] != seeds:
        raise ValueError("reference evaluation seeds differ")
    if reference["source_dataset_sha256"] != artifact["source_dataset_sha256"]:
        raise ValueError("reference and ACT checkpoint use different datasets")
    if reference["act_checkpoint_sha256"] != hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest():
        raise ValueError("reference and current ACT checkpoints differ")

    policies = {
        "latest_first_action": ACTLowDimPolicy(
            model, normalization, mode="latest_first_action"
        ),
        "temporal_ensemble": ACTLowDimPolicy(
            model,
            normalization,
            mode="temporal_ensemble",
            decay=float(config["temporal_decay"]),
        ),
    }
    environment = make_environment(
        config, read_json(PROJECT_ROOT / str(config["environment_config_path"]))
    )
    if environment.action_space.shape != (model.action_size,):
        environment.close()
        raise ValueError("ACT action dimension does not match the environment")
    failure_traces: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, Any]] = {}
    try:
        for delay in delays:
            policy_results = {
                name: evaluate(
                    environment,
                    policy,
                    seeds,
                    delay_steps=delay,
                    failure_traces=failure_traces,
                    policy_name=name,
                )
                for name, policy in policies.items()
            }
            targets = [
                [episode["target_position"] for episode in item["episodes"]]
                for item in policy_results.values()
            ]
            if targets[0] != targets[1]:
                raise ValueError("execution modes received different targets")
            results[str(delay)] = policy_results
    finally:
        environment.close()
    for name, previous_name in (
        ("latest_first_action", "act_latest_first_action"),
        ("temporal_ensemble", "act_temporal_ensemble"),
    ):
        verify_nominal_matches_reference(
            results["0"][name], reference["policy_results"][previous_name]
        )
    reference_targets = [
        item["target_position"]
        for item in reference["policy_results"]["act_latest_first_action"]["episodes"]
    ]
    for policy_results in results.values():
        for item in policy_results.values():
            if [r["target_position"] for r in item["episodes"]] != reference_targets:
                raise ValueError("delay conditions received different targets")

    output = {
        "experiment_name": config["experiment_name"],
        "status": "bounded_simulation_action_delay_stress_not_paper_reproduction",
        "source_dataset_sha256": artifact["source_dataset_sha256"],
        "act_checkpoint_sha256": reference["act_checkpoint_sha256"],
        "evaluation_seeds": seeds,
        "action_delay_steps": delays,
        "temporal_decay": float(config["temporal_decay"]),
        "nominal_matches_experiment_051": True,
        "conditions": results,
        "failure_trace_count": len(failure_traces) // 2,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    np.savez_compressed(OUTPUT_DIR / "failure_trajectories.npz", **failure_traces)
    for delay, policy_results in results.items():
        for name, item in policy_results.items():
            summary = item["summary"]
            print(
                f"delay={delay} {name}: "
                f"success={summary['success_count']}/{summary['episode_count']} "
                f"distance={summary['mean_final_distance']:.4f} "
                f"steps={summary['mean_step_count']:.1f} "
                f"overshoot={summary['mean_overshoot']:.4f}"
            )


if __name__ == "__main__":
    main()
