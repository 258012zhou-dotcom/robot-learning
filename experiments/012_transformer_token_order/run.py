"""Compare token-order Transformers with and without position embeddings."""

import json
import logging
from pathlib import Path
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_learning.dynamics_model import select_torch_device
from robot_learning.sequence_transformer import (
    TokenOrderTransformer,
    create_sequence_data_loader,
    evaluate_sequence_model,
    generate_token_order_dataset,
    train_sequence_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "012_transformer_token_order.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "012_transformer_token_order"


def load_config() -> dict[str, Any]:
    """Load experiment configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def create_model(config: dict[str, Any], use_positions: bool) -> TokenOrderTransformer:
    """Create one model from shared architecture settings."""
    return TokenOrderTransformer(
        vocabulary_size=int(config["vocabulary_size"]),
        maximum_token_count=int(config["sequence_length"]),
        embedding_dimension=int(config["embedding_dimension"]),
        head_count=int(config["head_count"]),
        mlp_hidden_dimension=int(config["mlp_hidden_dimension"]),
        layer_count=int(config["layer_count"]),
        use_position_embedding=use_positions,
    )


def save_learning_curves(results: dict[str, Any]) -> None:
    """Save training loss and validation accuracy comparisons."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, result in results.items():
        epochs = np.arange(1, len(result.training_losses) + 1)
        axes[0].plot(epochs, result.training_losses, label=name)
        axes[1].plot(epochs, result.validation_accuracies, label=name)
    axes[0].set(
        title="training cross-entropy",
        xlabel="epoch",
        ylabel="loss",
    )
    axes[1].set(
        title="validation accuracy",
        xlabel="epoch",
        ylabel="accuracy",
        ylim=(0.0, 1.05),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "learning_comparison.png", dpi=150)
    plt.close(figure)


def save_accuracy_plot(accuracies: dict[str, float]) -> None:
    """Save test accuracy with the balanced chance level."""
    names = list(accuracies)
    values = [accuracies[name] for name in names]
    figure, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(names, values, color=("gray", "tab:blue"))
    axis.axhline(0.5, color="black", linestyle="--", label="chance level")
    axis.set(
        title="Token-order test accuracy",
        ylabel="accuracy",
        ylim=(0.0, 1.05),
    )
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value * 100:.1f}%",
            ha="center",
        )
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "test_accuracy.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Train both controlled models and save comparison evidence."""
    config = load_config()
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

    dataset_arguments = {
        "sequence_length": int(config["sequence_length"]),
        "vocabulary_size": int(config["vocabulary_size"]),
    }
    train_dataset = generate_token_order_dataset(
        pair_count=int(config["train_pair_count"]),
        seed=seed,
        **dataset_arguments,
    )
    validation_dataset = generate_token_order_dataset(
        pair_count=int(config["validation_pair_count"]),
        seed=seed + 1,
        **dataset_arguments,
    )
    test_dataset = generate_token_order_dataset(
        pair_count=int(config["test_pair_count"]),
        seed=seed + 2,
        **dataset_arguments,
    )
    batch_size = int(config["batch_size"])
    validation_loader = create_sequence_data_loader(
        validation_dataset,
        batch_size,
        shuffle=False,
        seed=seed + 3,
    )
    test_loader = create_sequence_data_loader(
        test_dataset,
        batch_size,
        shuffle=False,
        seed=seed + 3,
    )
    device = select_torch_device()
    set_random_seeds(seed + 10)
    without_positions = create_model(config, use_positions=False)
    set_random_seeds(seed + 11)
    with_positions = create_model(config, use_positions=True)
    positioned_state = with_positions.state_dict()
    for name, parameter in without_positions.state_dict().items():
        positioned_state[name].copy_(parameter)
    with_positions.load_state_dict(positioned_state)
    models = {
        "without_positions": without_positions,
        "with_positions": with_positions,
    }
    training_results: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    for name, model in models.items():
        train_loader = create_sequence_data_loader(
            train_dataset,
            batch_size,
            shuffle=True,
            seed=seed + 4,
        )
        training = train_sequence_model(
            model,
            train_loader,
            validation_loader,
            device=device,
            epochs=int(config["epochs"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        evaluation = evaluate_sequence_model(model, test_loader, device)
        training_results[name] = training
        evaluations[name] = evaluation
        logger.info(
            "%s：best epoch %d，test accuracy %.2f%%，test loss %.4f",
            name,
            training.best_epoch,
            evaluation.accuracy * 100.0,
            evaluation.loss,
        )

    first_pair = test_dataset.token_ids[:2].to(device)
    paired_probabilities = {}
    with torch.no_grad():
        for name, model in models.items():
            probabilities = torch.softmax(model(first_pair), dim=1)[:, 1]
            paired_probabilities[name] = probabilities.cpu().tolist()

    accuracies = {
        name: evaluation.accuracy
        for name, evaluation in evaluations.items()
    }
    save_learning_curves(training_results)
    save_accuracy_plot(accuracies)
    results = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "device": str(device),
        "task": "label one when token A precedes token B",
        "sequence_length": int(config["sequence_length"]),
        "sample_counts": {
            "train": int(train_dataset.labels.numel()),
            "validation": int(validation_dataset.labels.numel()),
            "test": int(test_dataset.labels.numel()),
        },
        "methods": {
            name: {
                "best_epoch": training_results[name].best_epoch,
                "best_validation_loss": (
                    training_results[name].best_validation_loss
                ),
                "test_loss": evaluations[name].loss,
                "test_accuracy": evaluations[name].accuracy,
                "first_pair_positive_class_probabilities": (
                    paired_probabilities[name]
                ),
            }
            for name in models
        },
        "first_test_pair": {
            "positive_sequence": test_dataset.token_ids[0].tolist(),
            "negative_sequence": test_dataset.token_ids[1].tolist(),
            "labels": test_dataset.labels[:2].tolist(),
        },
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    logger.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
