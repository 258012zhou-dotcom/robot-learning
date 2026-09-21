"""Train BC with only target error while preserving experiment 027's budget."""

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from robot_learning.behavior_cloning import (
    BehaviorCloningMLP,
    prepare_behavior_cloning_data,
    save_behavior_cloning_checkpoint,
)
from robot_learning.training import create_data_loader, evaluate_mse, train_model
from robot_learning.trajectory_dataset import load_transition_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "029_bc_input_ablation.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "029_bc_input_ablation"


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    """Train the target-error-only model on unchanged expert rows."""
    config = load_json(CONFIG_PATH)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = load_transition_dataset(
        PROJECT_ROOT / str(config["dataset_path"])
    )
    manifest = load_json(
        PROJECT_ROOT / str(config["dataset_manifest_path"])
    )
    observation_indices = [int(index) for index in config["observation_indices"]]
    prepared = prepare_behavior_cloning_data(
        dataset,
        expert_policy_id=int(config["expert_policy_id"]),
        observation_indices=observation_indices,
    )
    loaders = {
        name: create_data_loader(
            torch.from_numpy(getattr(prepared, name).observations),
            torch.from_numpy(getattr(prepared, name).actions),
            batch_size=int(config["batch_size"]),
            shuffle=name == "train",
            seed=seed,
        )
        for name in ("train", "validation", "test")
    }

    action_low, action_high = manifest["action_range"]
    device = torch.device(str(config["device"]))
    if device.type != "cpu":
        raise ValueError("experiment 029 currently requires device='cpu'")
    model = BehaviorCloningMLP(
        observation_size=prepared.train.observations.shape[1],
        action_low=[float(action_low)],
        action_high=[float(action_high)],
        hidden_size=int(config["hidden_size"]),
    )
    initial_validation_mse = evaluate_mse(model, loaders["validation"], device)
    training_result = train_model(
        model,
        loaders["train"],
        loaders["validation"],
        device=device,
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    test_mse = evaluate_mse(model, loaders["test"], device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_behavior_cloning_checkpoint(
        OUTPUT_DIR / "best_model.pt",
        model,
        prepared.observation_normalization,
        source_dataset_sha256=str(manifest["dataset_content_sha256"]),
        seed=seed,
        observation_indices=observation_indices,
    )
    results = {
        "experiment_name": config["experiment_name"],
        "status": "offline_training_complete",
        "seed": seed,
        "observation_indices": observation_indices,
        "observation_feature_names": config["observation_feature_names"],
        "split_transition_counts": {
            name: int(getattr(prepared, name).actions.shape[0])
            for name in ("train", "validation", "test")
        },
        "split_episode_counts": {
            name: int(np.unique(getattr(prepared, name).episode_ids).size)
            for name in ("train", "validation", "test")
        },
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "initial_validation_mse": initial_validation_mse,
        "best_epoch": training_result.best_epoch,
        "best_validation_mse": training_result.best_validation_loss,
        "test_mse": test_mse,
        "training_mse_by_epoch": training_result.training_losses,
        "validation_mse_by_epoch": training_result.validation_losses,
    }
    with (OUTPUT_DIR / "offline_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(
        f"target_error_only: best_epoch={training_result.best_epoch}, "
        f"validation_mse={training_result.best_validation_loss:.8g}, "
        f"test_mse={test_mse:.8g}"
    )


if __name__ == "__main__":
    main()
