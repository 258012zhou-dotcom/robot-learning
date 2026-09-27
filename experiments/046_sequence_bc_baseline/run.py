"""Train a small chunked-BC model and compare closed-loop execution schedules."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from robot_learning.behavior_cloning import (
    BehaviorCloningPolicy,
    load_behavior_cloning_checkpoint,
)
from robot_learning.gymnasium_rollout import (
    Policy,
    calculate_position_overshoot,
    run_episode,
    summarize_episodes,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "046_sequence_bc_baseline.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "046_sequence_bc_baseline"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def make_loader(
    split: SequenceSplit, *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(split.observations),
            torch.from_numpy(split.action_chunks),
            torch.from_numpy(split.valid_mask),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def evaluate_offline(model: SequenceBCMLP, loader: DataLoader) -> tuple[float, float]:
    """Return full valid-slot MSE and directly comparable first-action MSE."""
    model.eval()
    total_squared_error = 0.0
    valid_component_count = 0
    first_squared_error = 0.0
    first_component_count = 0
    with torch.inference_mode():
        for observations, actions, valid_mask in loader:
            predictions = model(observations)
            squared = (predictions - actions).square()
            total_squared_error += float(
                (squared * valid_mask.unsqueeze(-1)).sum().item()
            )
            valid_component_count += int(valid_mask.sum()) * model.action_size
            first_squared_error += float(squared[:, 0].sum().item())
            first_component_count += int(observations.shape[0]) * model.action_size
    return (
        total_squared_error / valid_component_count,
        first_squared_error / first_component_count,
    )


def train(
    model: SequenceBCMLP,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        for observations, actions, valid_mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_chunk_mse(model(observations), actions, valid_mask)
            loss.backward()
            optimizer.step()
        training_loss, _ = evaluate_offline(model, train_loader)
        validation_loss, _ = evaluate_offline(model, validation_loader)
        train_losses.append(training_loss)
        validation_losses.append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation_chunk_mse": best_loss,
        "train_chunk_mse_by_epoch": train_losses,
        "validation_chunk_mse_by_epoch": validation_losses,
    }


def make_environment(
    config: dict[str, Any], environment_config: dict[str, Any]
) -> PointRobotReachEnv:
    """Recreate experiment 020's simulation without a new task definition."""
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


def evaluate_closed_loop(
    environment: PointRobotReachEnv,
    policy: Policy,
    seeds: list[int],
) -> dict[str, Any]:
    episodes = []
    records = []
    for seed in seeds:
        if isinstance(policy, SequenceBCPolicy):
            policy.reset()
        result = run_episode(environment, policy, seed=seed)
        episodes.append(result)
        records.append(
            {
                "seed": seed,
                "target_position": float(result.observations[0, 2]),
                "success": result.is_success,
                "step_count": result.step_count,
                "total_reward": result.total_reward,
                "final_distance": abs(float(result.observations[-1, 3])),
                "overshoot": calculate_position_overshoot(result.observations),
            }
        )
    summary = asdict(summarize_episodes(episodes))
    summary["mean_overshoot"] = float(
        np.mean([record["overshoot"] for record in records])
    )
    return {"summary": summary, "episodes": records}


def main() -> None:
    config = read_json(CONFIG_PATH)
    if str(config["device"]) != "cpu":
        raise ValueError("experiment 046 uses CPU only")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_transition_dataset(PROJECT_ROOT / str(config["dataset_path"]))
    manifest = read_json(PROJECT_ROOT / str(config["dataset_manifest_path"]))
    source_hash = str(manifest["dataset_content_sha256"])
    prepared = prepare_sequence_data(
        dataset,
        horizon=int(config["horizon"]),
        expert_policy_id=int(config["expert_policy_id"]),
    )
    loaders = {
        name: make_loader(
            getattr(prepared, name),
            batch_size=int(config["batch_size"]),
            shuffle=name == "train",
            seed=seed,
        )
        for name in ("train", "validation", "test")
    }
    action_low, action_high = manifest["action_range"]
    model = SequenceBCMLP(
        prepared.train.observations.shape[1],
        [float(action_low)],
        [float(action_high)],
        horizon=int(config["horizon"]),
        hidden_size=int(config["hidden_size"]),
    )
    initial_validation_mse, _ = evaluate_offline(model, loaders["validation"])
    history = train(
        model,
        loaders["train"],
        loaders["validation"],
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    save_sequence_checkpoint(
        OUTPUT_DIR / "best_model.pt",
        model,
        prepared.normalization,
        source_dataset_sha256=source_hash,
        seed=seed,
    )
    model, normalization, artifact = load_sequence_checkpoint(
        OUTPUT_DIR / "best_model.pt"
    )
    if artifact["source_dataset_sha256"] != source_hash:
        raise ValueError("sequence checkpoint source hash mismatch")
    test_chunk_mse, test_first_action_mse = evaluate_offline(
        model, loaders["test"]
    )
    offline = {
        "experiment_name": config["experiment_name"],
        "status": "sequence_bc_not_act",
        "source_dataset_sha256": source_hash,
        "seed": seed,
        "horizon": model.horizon,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "split_transition_counts": {
            name: int(getattr(prepared, name).observations.shape[0])
            for name in ("train", "validation", "test")
        },
        "split_episode_counts": {
            name: int(np.unique(getattr(prepared, name).episode_ids).size)
            for name in ("train", "validation", "test")
        },
        "initial_validation_chunk_mse": initial_validation_mse,
        **history,
        "test_chunk_mse": test_chunk_mse,
        "test_first_action_mse": test_first_action_mse,
    }
    with (OUTPUT_DIR / "offline_results.json").open("w", encoding="utf-8") as file:
        json.dump(offline, file, indent=2)

    bc_model, bc_norm, bc_artifact = load_behavior_cloning_checkpoint(
        PROJECT_ROOT / str(config["bc_checkpoint_path"])
    )
    if bc_artifact["source_dataset_sha256"] != source_hash:
        raise ValueError("single-step BC checkpoint source hash mismatch")
    first_seed = int(config["evaluation_seed_start"])
    episode_count = int(config["evaluation_episode_count"])
    if episode_count <= 0:
        raise ValueError("evaluation_episode_count must be positive")
    seeds = list(range(first_seed, first_seed + episode_count))
    if set(seeds) & set(dataset.environment_seeds.tolist()):
        raise ValueError("evaluation seeds overlap the demonstration dataset")
    environment_config = read_json(
        PROJECT_ROOT / str(config["environment_config_path"])
    )
    environment = make_environment(config, environment_config)
    try:
        policies: dict[str, Policy] = {
            "single_step_bc": BehaviorCloningPolicy(bc_model, bc_norm),
            "sequence_hold_chunk": SequenceBCPolicy(
                model, normalization, mode="hold_chunk"
            ),
            "sequence_replan_each_step": SequenceBCPolicy(
                model, normalization, mode="replan_each_step"
            ),
        }
        policy_results = {
            name: evaluate_closed_loop(environment, policy, seeds)
            for name, policy in policies.items()
        }
    finally:
        environment.close()
    targets = [
        [episode["target_position"] for episode in result["episodes"]]
        for result in policy_results.values()
    ]
    if any(target_list != targets[0] for target_list in targets[1:]):
        raise RuntimeError("policies received different evaluation targets")
    closed = {
        "experiment_name": config["experiment_name"],
        "status": "simulation_closed_loop_complete_sequence_bc_not_act",
        "evaluation_seeds": seeds,
        "training_seed_overlap_count": 0,
        "policy_results": policy_results,
    }
    with (OUTPUT_DIR / "closed_loop_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(closed, file, indent=2)
    print(
        f"best_epoch={history['best_epoch']} "
        f"test_chunk_mse={test_chunk_mse:.8g} "
        f"test_first_action_mse={test_first_action_mse:.8g}"
    )
    for name, result in policy_results.items():
        summary = result["summary"]
        print(
            f"{name}: success={summary['success_count']}/{summary['episode_count']} "
            f"mean_final_distance={summary['mean_final_distance']:.4f} "
            f"mean_steps={summary['mean_step_count']:.1f}"
        )


if __name__ == "__main__":
    main()
