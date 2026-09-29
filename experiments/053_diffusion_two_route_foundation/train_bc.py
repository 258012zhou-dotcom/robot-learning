"""Train a one-seed deterministic chunk BC baseline on both expert routes."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from robot_learning.gymnasium_rollout import run_episode
from robot_learning.sequence_behavior_cloning import (
    SequenceBCMLP,
    SequenceBCPolicy,
    SequenceSplit,
    load_sequence_checkpoint,
    masked_chunk_mse,
    prepare_sequence_data,
    save_sequence_checkpoint,
)
from robot_learning.trajectory_dataset import load_transition_dataset
from robot_learning.two_route_navigation import TwoRouteNavigationEnv


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/053_diffusion_two_route_foundation"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def loader(split: SequenceSplit, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(split.observations),
            torch.from_numpy(split.action_chunks),
            torch.from_numpy(split.valid_mask),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def offline_mse(model: SequenceBCMLP, data: DataLoader) -> dict[str, float]:
    model.eval()
    chunk_error = 0.0
    valid_components = 0
    first_error = 0.0
    first_components = 0
    with torch.inference_mode():
        for observations, actions, mask in data:
            squared = (model(observations) - actions).square()
            chunk_error += float((squared * mask.unsqueeze(-1)).sum())
            valid_components += int(mask.sum()) * model.action_size
            first_error += float(squared[:, 0].sum())
            first_components += observations.shape[0] * model.action_size
    return {
        "chunk_mse": chunk_error / valid_components,
        "first_action_mse": first_error / first_components,
    }


def train(model: SequenceBCMLP, train_data: DataLoader, validation_data: DataLoader,
          config: dict) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        for observations, actions, mask in train_data:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_chunk_mse(model(observations), actions, mask)
            loss.backward()
            optimizer.step()
        validation_loss = offline_mse(model, validation_data)["chunk_mse"]
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_validation_chunk_mse": best_loss}


def evaluate_closed_loop(policy: SequenceBCPolicy, seeds: list[int], task: dict) -> dict:
    environment = TwoRouteNavigationEnv(
        dt=float(task["dt"]),
        max_episode_steps=int(task["max_episode_steps"]),
        goal_tolerance=float(task["goal_tolerance"]),
    )
    records = []
    try:
        for seed in seeds:
            policy.reset()
            result = run_episode(environment, policy, seed=seed)
            start = result.observations[0]
            records.append({
                "scene_seed": seed,
                "success": bool(result.is_success),
                "collision": bool(result.terminated and not result.is_success),
                "timeout": bool(result.truncated),
                "steps": result.step_count,
                "first_action": result.actions[0].tolist(),
                "last_position": result.observations[-1, :2].tolist(),
                "obstacle_center": start[4:6].tolist(),
                "obstacle_radius": float(start[6]),
                "final_distance": float(np.linalg.norm(result.observations[-1, :2] - start[2:4])),
                "y_at_obstacle_x": float(result.observations[np.argmin(
                    np.abs(result.observations[:, 0] - start[4])), 1]),
            })
    finally:
        environment.close()
    return {
        "scene_count": len(records),
        "success_count": sum(record["success"] for record in records),
        "collision_count": sum(record["collision"] for record in records),
        "timeout_count": sum(record["timeout"] for record in records),
        "mean_final_distance": float(np.mean([record["final_distance"] for record in records])),
        "episodes": records,
    }


def main() -> None:
    config = read_json(ROOT / "configs/053_diffusion_two_route_bc.json")
    task = read_json(ROOT / "configs/053_diffusion_two_route_foundation.json")
    foundation = read_json(OUTPUT_DIR / "results.json")
    dataset_path = OUTPUT_DIR / "dataset.npz"
    source_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if source_hash != foundation["dataset_sha256"]:
        raise RuntimeError("dataset changed since expert audit")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    dataset = load_transition_dataset(dataset_path)
    prepared = prepare_sequence_data(
        dataset, horizon=int(config["prediction_horizon"]), expert_policy_ids=[0, 1]
    )
    datasets = {
        name: loader(getattr(prepared, name), batch_size=int(config["batch_size"]),
                     shuffle=name == "train", seed=seed)
        for name in ("train", "validation", "test")
    }
    model = SequenceBCMLP(
        7, [-1.0, -1.0], [1.0, 1.0],
        horizon=int(config["prediction_horizon"]),
        hidden_size=int(config["hidden_size"]),
    )
    initial_validation = offline_mse(model, datasets["validation"])
    history = train(model, datasets["train"], datasets["validation"], config)
    checkpoint_path = OUTPUT_DIR / "bc_best_model.pt"
    save_sequence_checkpoint(checkpoint_path, model, prepared.normalization,
                             source_dataset_sha256=source_hash, seed=seed)
    model, normalization, artifact = load_sequence_checkpoint(checkpoint_path)
    if artifact["source_dataset_sha256"] != source_hash:
        raise RuntimeError("checkpoint and dataset identity disagree")
    policy = SequenceBCPolicy(
        model, normalization, mode="hold_chunk",
        execution_horizon=int(config["execution_horizon"]),
    )
    validation_seeds = list(range(
        int(task["scene_splits"]["validation"]["seed_start"]),
        int(task["scene_splits"]["validation"]["seed_start"])
        + int(task["scene_splits"]["validation"]["scene_count"]),
    ))
    test_seeds = list(range(
        int(task["scene_splits"]["test"]["seed_start"]),
        int(task["scene_splits"]["test"]["seed_start"])
        + int(task["scene_splits"]["test"]["scene_count"]),
    ))
    report = {
        "status": "deterministic_chunk_bc_one_seed_pilot_not_diffusion_policy",
        "source_dataset_sha256": source_hash,
        "config": config,
        "split_transition_counts": {
            name: len(getattr(prepared, name).observations)
            for name in datasets
        },
        "initial_validation_offline": initial_validation,
        "training": history,
        "train_offline": offline_mse(model, datasets["train"]),
        "validation_offline": offline_mse(model, datasets["validation"]),
        "test_offline": offline_mse(model, datasets["test"]),
        "validation_closed_loop": evaluate_closed_loop(policy, validation_seeds, task),
        "test_closed_loop": evaluate_closed_loop(policy, test_seeds, task),
    }
    with (OUTPUT_DIR / "bc_results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        "best_epoch": history["best_epoch"],
        "test_offline": report["test_offline"],
        "validation_success": report["validation_closed_loop"]["success_count"],
        "test_success": report["test_closed_loop"]["success_count"],
        "test_collisions": report["test_closed_loop"]["collision_count"],
        "test_timeouts": report["test_closed_loop"]["timeout_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
