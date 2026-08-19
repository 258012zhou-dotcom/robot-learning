"""用小样本和带噪标签观察过拟合及权重衰减。"""

from copy import deepcopy
import json
import logging
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_learning.dynamics_model import (
    DynamicsMLP,
    select_torch_device,
)
from robot_learning.motion_dataset import (
    fit_input_standardization,
    generate_motion_samples,
    standardize_motion_inputs,
)
from robot_learning.training import create_data_loader, train_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "004_learned_dynamics.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "004_learned_dynamics"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    config = load_config()
    demo_config = config["overfitting_demo"]
    seed = int(config["seed"])
    train_count = int(demo_config["train_sample_count"])
    validation_count = int(demo_config["validation_sample_count"])
    sample_count = train_count + validation_count
    set_random_seeds(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    inputs, clean_targets = generate_motion_samples(
        sample_count=sample_count,
        dt=float(config["dt"]),
        seed=seed,
    )
    train_inputs = inputs[:train_count]
    validation_inputs = inputs[train_count:]
    clean_train_targets = clean_targets[:train_count]
    validation_targets = clean_targets[train_count:]

    noise_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    noisy_train_targets = clean_train_targets + torch.randn(
        clean_train_targets.shape,
        generator=noise_generator,
    ) * float(demo_config["label_noise_std"])

    stats = fit_input_standardization(train_inputs)
    standardized_train_inputs = standardize_motion_inputs(
        train_inputs,
        stats,
    )
    standardized_validation_inputs = standardize_motion_inputs(
        validation_inputs,
        stats,
    )

    def make_loaders():
        train_loader = create_data_loader(
            standardized_train_inputs,
            noisy_train_targets,
            batch_size=int(demo_config["batch_size"]),
            shuffle=True,
            seed=seed,
        )
        validation_loader = create_data_loader(
            standardized_validation_inputs,
            validation_targets,
            batch_size=int(demo_config["batch_size"]),
            shuffle=False,
            seed=seed,
        )
        return train_loader, validation_loader

    hidden_size = int(demo_config["hidden_size"])
    set_random_seeds(seed)
    initial_model = DynamicsMLP(hidden_size=hidden_size)
    initial_state = deepcopy(initial_model.state_dict())
    device = select_torch_device()

    comparisons = {}
    histories = {}
    for name, weight_decay in (
        ("no_weight_decay", 0.0),
        ("with_weight_decay", float(demo_config["weight_decay"])),
    ):
        model = DynamicsMLP(hidden_size=hidden_size)
        model.load_state_dict(initial_state)
        train_loader, validation_loader = make_loaders()
        result = train_model(
            model,
            train_loader,
            validation_loader,
            device=device,
            epochs=int(demo_config["epochs"]),
            learning_rate=float(demo_config["learning_rate"]),
            weight_decay=weight_decay,
        )
        comparisons[name] = {
            "weight_decay": weight_decay,
            "best_epoch": result.best_epoch,
            "best_validation_mse": result.best_validation_loss,
            "final_training_mse": result.training_losses[-1],
            "final_validation_mse": result.validation_losses[-1],
            "validation_increase_after_best": (
                result.validation_losses[-1]
                - result.best_validation_loss
            ),
        }
        histories[name] = result

    results = {
        "experiment_name": "004_learned_dynamics_overfitting_demo",
        "seed": seed,
        "device": str(device),
        "train_sample_count": train_count,
        "validation_sample_count": validation_count,
        "label_noise_std": float(demo_config["label_noise_std"]),
        "hidden_size": hidden_size,
        "parameter_count": sum(
            parameter.numel() for parameter in initial_model.parameters()
        ),
        "comparisons": comparisons,
    }
    with (OUTPUT_DIR / "overfitting_results.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, (name, result) in zip(axes, histories.items()):
        epoch_numbers = np.arange(1, len(result.training_losses) + 1)
        axis.plot(
            epoch_numbers,
            result.training_losses,
            label="training MSE",
        )
        axis.plot(
            epoch_numbers,
            result.validation_losses,
            label="validation MSE",
        )
        axis.axvline(
            result.best_epoch,
            color="gray",
            linestyle="--",
            label="best epoch",
        )
        axis.set(
            title=name.replace("_", " "),
            xlabel="epoch",
            ylabel="MSE",
            yscale="log",
        )
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "overfitting_comparison.png",
        dpi=150,
    )
    plt.close(figure)

    logger.info("设备：%s", device)
    logger.info("模型参数量：%s", results["parameter_count"])
    for name, metrics in comparisons.items():
        logger.info(
            "%s：最佳轮次 %s，最佳验证 MSE %.6f，"
            "最终训练 MSE %.6f，最终验证 MSE %.6f",
            name,
            metrics["best_epoch"],
            metrics["best_validation_mse"],
            metrics["final_training_mse"],
            metrics["final_validation_mse"],
        )


if __name__ == "__main__":
    main()
