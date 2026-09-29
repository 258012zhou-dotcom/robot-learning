"""Compare instructed and uninstructed chunk BC on the same two-route scenes."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from robot_learning.gymnasium_rollout import run_episode
from robot_learning.route_conditioning import (
    RouteConditionedSequenceBCPolicy,
    prepare_route_conditioned_data,
    route_at_obstacle_x,
)
from robot_learning.sequence_behavior_cloning import (
    SequenceBCMLP,
    SequenceBCPolicy,
    SequenceSplit,
    load_sequence_checkpoint,
    masked_chunk_mse,
    save_sequence_checkpoint,
)
from robot_learning.trajectory_dataset import load_transition_dataset
from robot_learning.two_route_navigation import TwoRouteNavigationEnv


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/054_route_conditioning.json"
OUTPUT_DIR = ROOT / "outputs/054_route_conditioning"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def make_loader(
    split: SequenceSplit, *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
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


def chunk_mse(model: SequenceBCMLP, data: DataLoader) -> float:
    model.eval()
    squared_error = 0.0
    component_count = 0
    with torch.inference_mode():
        for observations, actions, mask in data:
            squared = (model(observations) - actions).square()
            squared_error += float((squared * mask.unsqueeze(-1)).sum())
            component_count += int(mask.sum()) * model.action_size
    return squared_error / component_count


def train(
    model: SequenceBCMLP, training: DataLoader, validation: DataLoader, config: dict
) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        for observations, actions, mask in training:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_chunk_mse(model(observations), actions, mask)
            loss.backward()
            optimizer.step()
        validation_loss = chunk_mse(model, validation)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_validation_chunk_mse": best_loss}


def scene_seeds(task: dict, split_name: str) -> list[int]:
    settings = task["scene_splits"][split_name]
    start = int(settings["seed_start"])
    count = int(settings["scene_count"])
    return list(range(start, start + count))


def evaluate(
    environment: TwoRouteNavigationEnv,
    policy: SequenceBCPolicy | RouteConditionedSequenceBCPolicy,
    seeds: list[int],
) -> dict:
    records = []
    for seed in seeds:
        paired_results = []
        for requested_route in (-1, 1):
            if isinstance(policy, RouteConditionedSequenceBCPolicy):
                policy.reset(route=requested_route)
            else:
                policy.reset()
            result = run_episode(environment, policy, seed=seed)
            paired_results.append(result)
            actual_route = route_at_obstacle_x(result.observations)
            matched = actual_route == requested_route
            records.append({
                "scene_seed": seed,
                "requested_route": requested_route,
                "actual_route": actual_route,
                "task_success": bool(result.is_success),
                "route_match": matched,
                "joint_success": bool(result.is_success and matched),
                "collision": bool(result.terminated and not result.is_success),
                "timeout": bool(result.truncated),
                "step_count": result.step_count,
                "first_action": result.actions[0].tolist(),
            })
        if isinstance(policy, SequenceBCPolicy):
            lower, upper = records[-2:]
            if (lower["first_action"] != upper["first_action"] or
                    lower["task_success"] != upper["task_success"] or
                    lower["actual_route"] != upper["actual_route"] or
                    not np.array_equal(paired_results[0].observations,
                                       paired_results[1].observations) or
                    not np.array_equal(paired_results[0].actions,
                                       paired_results[1].actions)):
                raise RuntimeError("uninstructed deterministic baseline changed with request")
        elif not np.array_equal(
            paired_results[0].observations[0], paired_results[1].observations[0]
        ):
            raise RuntimeError("paired route requests did not start from the same scene")
    return {
        "scene_count": len(seeds),
        "request_count": len(records),
        "task_success_count": sum(row["task_success"] for row in records),
        "route_match_count": sum(row["route_match"] for row in records),
        "joint_success_count": sum(row["joint_success"] for row in records),
        "collision_count": sum(row["collision"] for row in records),
        "timeout_count": sum(row["timeout"] for row in records),
        "requests": records,
    }


def main() -> None:
    config = read_json(CONFIG_PATH)
    bc_config = read_json(ROOT / config["baseline_bc_config_path"])
    task = read_json(ROOT / config["source_task_config_path"])
    manifest = read_json(ROOT / config["source_manifest_path"])
    dataset_path = ROOT / config["source_dataset_path"]
    source_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if source_hash != manifest["dataset_sha256"]:
        raise RuntimeError("dataset changed since source audit")
    seed = int(bc_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    dataset = load_transition_dataset(dataset_path)
    prepared = prepare_route_conditioned_data(
        dataset, horizon=int(bc_config["prediction_horizon"])
    )
    loaders = {
        name: make_loader(
            getattr(prepared, name), batch_size=int(bc_config["batch_size"]),
            shuffle=name == "train", seed=seed,
        )
        for name in ("train", "validation", "test")
    }
    model = SequenceBCMLP(
        observation_size=8,
        action_low=[-1.0, -1.0],
        action_high=[1.0, 1.0],
        horizon=int(bc_config["prediction_horizon"]),
        hidden_size=int(bc_config["hidden_size"]),
    )
    initial_validation_mse = chunk_mse(model, loaders["validation"])
    history = train(model, loaders["train"], loaders["validation"], bc_config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conditioned_path = OUTPUT_DIR / "conditioned_bc_model.pt"
    save_sequence_checkpoint(
        conditioned_path, model, prepared.normalization,
        source_dataset_sha256=source_hash, seed=seed,
    )
    model, normalization, artifact = load_sequence_checkpoint(conditioned_path)
    if artifact["source_dataset_sha256"] != source_hash:
        raise RuntimeError("conditioned checkpoint dataset identity mismatch")
    baseline_model, baseline_normalization, baseline_artifact = load_sequence_checkpoint(
        ROOT / config["baseline_checkpoint_path"]
    )
    if (baseline_artifact["source_dataset_sha256"] != source_hash or
            baseline_artifact["seed"] != seed or
            baseline_model.observation_size != 7 or
            baseline_model.horizon != model.horizon or
            baseline_model.hidden_size != model.hidden_size):
        raise RuntimeError("baseline model is not the frozen 053 BC contract")
    conditioned_policy = RouteConditionedSequenceBCPolicy(
        model, normalization, execution_horizon=int(bc_config["execution_horizon"])
    )
    baseline_policy = SequenceBCPolicy(
        baseline_model, baseline_normalization, mode="hold_chunk",
        execution_horizon=int(bc_config["execution_horizon"]),
    )
    environment = TwoRouteNavigationEnv(
        dt=float(task["dt"]),
        max_episode_steps=int(task["max_episode_steps"]),
        goal_tolerance=float(task["goal_tolerance"]),
    )
    try:
        comparisons = {
            split: {
                "conditioned_bc": evaluate(
                    environment, conditioned_policy, scene_seeds(task, split)
                ),
                "uninstructed_bc": evaluate(
                    environment, baseline_policy, scene_seeds(task, split)
                ),
            }
            for split in ("validation", "test")
        }
    finally:
        environment.close()
    report = {
        "status": "one_seed_synthetic_route_instruction_teaching_experiment",
        "source_dataset_sha256": source_hash,
        "seed": seed,
        "bc_training_config": bc_config,
        "split_transition_counts": {
            name: len(getattr(prepared, name).observations) for name in loaders
        },
        "initial_validation_chunk_mse": initial_validation_mse,
        "training": history,
        "test_offline_chunk_mse": chunk_mse(model, loaders["test"]),
        "comparisons": comparisons,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        "best_epoch": history["best_epoch"],
        "test_offline_chunk_mse": report["test_offline_chunk_mse"],
        "test_conditioned": {
            key: value for key, value in comparisons["test"]["conditioned_bc"].items()
            if key != "requests"
        },
        "test_uninstructed": {
            key: value for key, value in comparisons["test"]["uninstructed_bc"].items()
            if key != "requests"
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
