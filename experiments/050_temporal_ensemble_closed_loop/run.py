"""Compare two execution schedules with one frozen sequence-BC checkpoint."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_learning.gymnasium_rollout import (
    calculate_position_overshoot,
    run_episode,
    summarize_episodes,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.sequence_behavior_cloning import (
    SequenceBCPolicy,
    SequenceBCTemporalEnsemblePolicy,
    load_sequence_checkpoint,
)
from robot_learning.trajectory_dataset import load_transition_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "050_temporal_ensemble_closed_loop.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "050_temporal_ensemble_closed_loop"
SequenceExecutionPolicy = SequenceBCPolicy | SequenceBCTemporalEnsemblePolicy


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
    policy: SequenceExecutionPolicy,
    seeds: list[int],
) -> dict[str, Any]:
    episodes = []
    records = []
    for seed in seeds:
        policy.reset()
        result = run_episode(environment, policy, seed=seed)
        episodes.append(result)
        action_changes = np.abs(np.diff(result.actions, axis=0))
        records.append(
            {
                "seed": seed,
                "target_position": float(result.observations[0, 2]),
                "success": result.is_success,
                "step_count": result.step_count,
                "total_reward": result.total_reward,
                "final_distance": abs(float(result.observations[-1, 3])),
                "overshoot": calculate_position_overshoot(result.observations),
                "mean_abs_command_change": float(action_changes.mean())
                if action_changes.size
                else 0.0,
            }
        )
    summary = asdict(summarize_episodes(episodes))
    summary["mean_overshoot"] = float(np.mean([r["overshoot"] for r in records]))
    summary["mean_abs_command_change"] = float(
        np.mean([r["mean_abs_command_change"] for r in records])
    )
    return {"summary": summary, "episodes": records}


def check_previous_baseline(
    current: dict[str, Any], previous: dict[str, Any], seeds: list[int]
) -> None:
    if previous["evaluation_seeds"] != seeds:
        raise ValueError("previous evaluation seeds differ")
    old_records = previous["policy_results"]["sequence_replan_each_step"]["episodes"]
    for new, old in zip(current["episodes"], old_records, strict=True):
        if new["seed"] != old["seed"] or new["success"] != old["success"]:
            raise ValueError("replanned baseline no longer matches experiment 046")
        if new["step_count"] != old["step_count"]:
            raise ValueError("replanned baseline step count changed")
        for key in ("target_position", "final_distance", "total_reward"):
            if not np.isclose(new[key], old[key], rtol=0.0, atol=1e-6):
                raise ValueError(f"replanned baseline {key} changed")


def main() -> None:
    config = read_json(CONFIG_PATH)
    if config["device"] != "cpu":
        raise ValueError("this teaching comparison uses CPU only")
    torch.set_num_threads(1)
    checkpoint_path = PROJECT_ROOT / str(config["sequence_checkpoint_path"])
    model, normalization, artifact = load_sequence_checkpoint(checkpoint_path)
    manifest = read_json(PROJECT_ROOT / str(config["dataset_manifest_path"]))
    if artifact["source_dataset_sha256"] != manifest["dataset_content_sha256"]:
        raise ValueError("checkpoint and source dataset identity differ")
    dataset = load_transition_dataset(PROJECT_ROOT / str(config["dataset_path"]))
    first_seed = int(config["evaluation_seed_start"])
    episode_count = int(config["evaluation_episode_count"])
    if episode_count <= 0:
        raise ValueError("evaluation_episode_count must be positive")
    seeds = list(range(first_seed, first_seed + episode_count))
    if set(seeds) & set(dataset.environment_seeds.tolist()):
        raise ValueError("evaluation seeds overlap demonstration data")

    policies: dict[str, SequenceExecutionPolicy] = {
        "replan_latest_first_action": SequenceBCPolicy(
            model, normalization, mode="replan_each_step"
        ),
        "temporal_ensemble": SequenceBCTemporalEnsemblePolicy(
            model, normalization, decay=float(config["temporal_decay"])
        ),
    }
    environment = make_environment(
        config, read_json(PROJECT_ROOT / str(config["environment_config_path"]))
    )
    try:
        results = {
            name: evaluate(environment, policy, seeds)
            for name, policy in policies.items()
        }
    finally:
        environment.close()
    check_previous_baseline(
        results["replan_latest_first_action"],
        read_json(PROJECT_ROOT / str(config["previous_closed_loop_path"])),
        seeds,
    )
    targets = [
        [episode["target_position"] for episode in result["episodes"]]
        for result in results.values()
    ]
    if targets[0] != targets[1]:
        raise ValueError("policies received different evaluation targets")
    output = {
        "experiment_name": config["experiment_name"],
        "status": "simulation_closed_loop_non_act_sequence_bc",
        "source_dataset_sha256": artifact["source_dataset_sha256"],
        "sequence_checkpoint_sha256": hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest(),
        "evaluation_seeds": seeds,
        "temporal_decay": float(config["temporal_decay"]),
        "previous_baseline_matches": True,
        "policy_results": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "closed_loop_results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    for name, result in results.items():
        summary = result["summary"]
        print(
            f"{name}: success={summary['success_count']}/{summary['episode_count']} "
            f"distance={summary['mean_final_distance']:.5f} "
            f"steps={summary['mean_step_count']:.1f} "
            f"command_change={summary['mean_abs_command_change']:.6f}"
        )


if __name__ == "__main__":
    main()
