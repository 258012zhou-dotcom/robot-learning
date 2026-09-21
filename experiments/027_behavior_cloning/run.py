"""Train and evaluate the offline part of the Behavior Cloning baseline."""

import json
import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from robot_learning.behavior_cloning import (
    BehaviorCloningData,
    BehaviorCloningMLP,
    prepare_behavior_cloning_data,
    save_behavior_cloning_checkpoint,
)
from robot_learning.training import (
    create_data_loader,
    evaluate_mse,
    train_model,
)
from robot_learning.trajectory_dataset import load_transition_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "027_behavior_cloning.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "027_behavior_cloning"


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for this small CPU experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_loaders(
    data: BehaviorCloningData,
    *,
    batch_size: int,
    seed: int,
) -> dict[str, torch.utils.data.DataLoader]:
    """Build a shuffled train loader and deterministic evaluation loaders."""
    loaders: dict[str, torch.utils.data.DataLoader] = {}
    for name in ("train", "validation", "test"):
        split = getattr(data, name)
        loaders[name] = create_data_loader(
            torch.from_numpy(split.observations),
            torch.from_numpy(split.actions),
            batch_size=batch_size,
            shuffle=name == "train",
            seed=seed,
        )
    return loaders


def main() -> None:
    """Train with expert transitions and reserve test data for final MSE."""
    config = load_json(CONFIG_PATH)
    seed = int(config["seed"])
    set_random_seeds(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "run.log", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)

    dataset_path = PROJECT_ROOT / str(config["dataset_path"])
    manifest_path = PROJECT_ROOT / str(config["dataset_manifest_path"])
    dataset = load_transition_dataset(dataset_path)
    manifest = load_json(manifest_path)
    prepared = prepare_behavior_cloning_data(
        dataset,
        expert_policy_id=int(config["expert_policy_id"]),
    )
    action_low, action_high = manifest["action_range"]
    loaders = create_loaders(
        prepared,
        batch_size=int(config["batch_size"]),
        seed=seed,
    )

    device = torch.device(str(config["device"]))
    if device.type != "cpu":
        raise ValueError("experiment 027 currently requires device='cpu'")
    model = BehaviorCloningMLP(
        observation_size=prepared.train.observations.shape[1],
        action_low=[float(action_low)],
        action_high=[float(action_high)],
        hidden_size=int(config["hidden_size"]),
    )
    initial_validation_mse = evaluate_mse(
        model, loaders["validation"], device
    )
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

    save_behavior_cloning_checkpoint(
        OUTPUT_DIR / "best_model.pt",
        model,
        prepared.observation_normalization,
        source_dataset_sha256=str(manifest["dataset_content_sha256"]),
        seed=seed,
    )

    results = {
        "experiment_name": config["experiment_name"],
        "status": "offline_training_complete_closed_loop_pending",
        "seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "source_dataset_sha256": manifest["dataset_content_sha256"],
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

    logger.info(
        "offline training complete: best_epoch=%d, validation_mse=%.8g, "
        "test_mse=%.8g",
        training_result.best_epoch,
        training_result.best_validation_loss,
        test_mse,
    )


if __name__ == "__main__":
    main()
