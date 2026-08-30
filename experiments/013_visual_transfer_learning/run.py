"""Compare scratch, frozen probe, and full visual fine-tuning."""

from copy import deepcopy
import json
import logging
from pathlib import Path
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_learning.dynamics_model import select_torch_device
from robot_learning.image_classification import (
    create_classification_data_loader,
    evaluate_classifier,
    generate_shape_challenge_dataset,
    generate_shape_classification_dataset,
    split_classification_dataset,
)
from robot_learning.transfer_learning import (
    create_source_head_evaluator,
    set_encoder_trainable,
    train_transfer_classifier,
)
from robot_learning.visual_representation import EncoderClassifier, SmallVisualEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "013_visual_transfer_learning.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "013_visual_transfer_learning"
METHOD_NAMES = ("scratch", "linear_probe", "full_finetune")


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


def create_loader(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    """Create a classification loader with shared options."""
    return create_classification_data_loader(
        images,
        labels,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def aggregate_trial_values(
    trials: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, dict[str, float]]:
    """Return mean and population standard deviation per method."""
    aggregated = {}
    for method_name in METHOD_NAMES:
        values = np.array(
            [trial[method_name][metric_name] for trial in trials],
            dtype=np.float64,
        )
        aggregated[method_name] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=0)),
        }
    return aggregated


def save_accuracy_comparison(
    target_aggregate: dict[str, dict[str, float]],
    source_aggregate: dict[str, dict[str, float]],
    pretrained_source_accuracy: float,
) -> None:
    """Save target transfer and source-domain retention accuracy."""
    positions = np.arange(len(METHOD_NAMES))
    target_means = [target_aggregate[name]["mean"] for name in METHOD_NAMES]
    target_errors = [
        target_aggregate[name]["standard_deviation"] for name in METHOD_NAMES
    ]
    source_means = [source_aggregate[name]["mean"] for name in METHOD_NAMES]
    source_errors = [
        source_aggregate[name]["standard_deviation"] for name in METHOD_NAMES
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(positions, target_means, yerr=target_errors, capsize=5)
    axes[0].set(title="rotated target test", ylabel="accuracy")
    axes[1].bar(positions, source_means, yerr=source_errors, capsize=5)
    axes[1].axhline(
        pretrained_source_accuracy,
        color="black",
        linestyle="--",
        label="pretrained source baseline",
    )
    axes[1].set(
        title="adapted encoder + original source head",
        ylabel="accuracy",
    )
    axes[1].legend()
    for axis in axes:
        axis.set(
            xticks=positions,
            xticklabels=METHOD_NAMES,
            ylim=(0.0, 1.05),
        )
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(figure)


def save_first_trial_curves(training_results: dict[str, Any]) -> None:
    """Save target training loss and validation accuracy for the first trial."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method_name, training in training_results.items():
        epochs = np.arange(1, len(training.training_losses) + 1)
        axes[0].plot(epochs, training.training_losses, label=method_name)
        axes[1].plot(epochs, training.validation_accuracies, label=method_name)
    axes[0].set(title="target training loss", xlabel="epoch", ylabel="loss")
    axes[1].set(
        title="target validation accuracy",
        xlabel="epoch",
        ylabel="accuracy",
        ylim=(0.0, 1.05),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "first_trial_learning_curves.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Pretrain once, adapt three ways, and evaluate repeated target trials."""
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
    device = select_torch_device()
    batch_size = int(config["batch_size"])

    source_images, source_labels = generate_shape_classification_dataset(
        int(config["source_sample_count"]),
        int(config["image_size"]),
        seed,
    )
    source_splits = split_classification_dataset(
        source_images,
        source_labels,
        float(config["source_train_fraction"]),
        float(config["source_validation_fraction"]),
        seed + 1,
    )
    source_train_loader = create_loader(
        source_splits.train_images,
        source_splits.train_labels,
        batch_size,
        True,
        seed + 2,
    )
    source_validation_loader = create_loader(
        source_splits.validation_images,
        source_splits.validation_labels,
        batch_size,
        False,
        seed + 2,
    )
    source_test_loader = create_loader(
        source_splits.test_images,
        source_splits.test_labels,
        batch_size,
        False,
        seed + 2,
    )
    set_random_seeds(seed + 10)
    pretrained_model = EncoderClassifier(
        SmallVisualEncoder(int(config["embedding_dimension"])),
        num_classes=2,
    )
    source_training = train_transfer_classifier(
        pretrained_model,
        source_train_loader,
        source_validation_loader,
        device,
        int(config["source_epochs"]),
        float(config["source_learning_rate"]),
        float(config["source_learning_rate"]),
        float(config["weight_decay"]),
        2,
    )
    pretrained_source_evaluation = evaluate_classifier(
        pretrained_model,
        source_test_loader,
        device,
        2,
    )
    target_test_images, target_test_labels = generate_shape_challenge_dataset(
        int(config["target_test_sample_count"]),
        int(config["image_size"]),
        seed + 3000,
    )
    target_test_loader = create_loader(
        target_test_images,
        target_test_labels,
        batch_size,
        False,
        seed + 3,
    )
    zero_shot_target = evaluate_classifier(
        pretrained_model,
        target_test_loader,
        device,
        2,
    )
    logger.info(
        "预训练：source %.2f%%，rotated zero-shot %.2f%%",
        pretrained_source_evaluation.accuracy * 100.0,
        zero_shot_target.accuracy * 100.0,
    )

    trial_results: list[dict[str, Any]] = []
    first_trial_training = {}
    for trial_index in range(int(config["trial_count"])):
        target_train_images, target_train_labels = generate_shape_challenge_dataset(
            int(config["target_train_sample_count"]),
            int(config["image_size"]),
            seed + 1000 + trial_index,
        )
        target_validation_images, target_validation_labels = (
            generate_shape_challenge_dataset(
                int(config["target_validation_sample_count"]),
                int(config["image_size"]),
                seed + 2000 + trial_index,
            )
        )
        train_loader = create_loader(
            target_train_images,
            target_train_labels,
            batch_size,
            True,
            seed + 100 + trial_index,
        )
        validation_loader = create_loader(
            target_validation_images,
            target_validation_labels,
            batch_size,
            False,
            seed + 100 + trial_index,
        )

        set_random_seeds(seed + 100 + trial_index)
        scratch = EncoderClassifier(
            SmallVisualEncoder(int(config["embedding_dimension"])),
            num_classes=2,
        )
        set_random_seeds(seed + 200 + trial_index)
        transfer_base = EncoderClassifier(
            deepcopy(pretrained_model.encoder),
            num_classes=2,
        )
        linear_probe = deepcopy(transfer_base)
        full_finetune = deepcopy(transfer_base)
        set_encoder_trainable(linear_probe, False)
        models = {
            "scratch": scratch,
            "linear_probe": linear_probe,
            "full_finetune": full_finetune,
        }
        learning_rates = {
            "scratch": (
                float(config["scratch_learning_rate"]),
                float(config["scratch_learning_rate"]),
            ),
            "linear_probe": (
                float(config["encoder_finetune_learning_rate"]),
                float(config["head_learning_rate"]),
            ),
            "full_finetune": (
                float(config["encoder_finetune_learning_rate"]),
                float(config["head_learning_rate"]),
            ),
        }
        current_trial = {}
        for method_name, model in models.items():
            encoder_lr, head_lr = learning_rates[method_name]
            training = train_transfer_classifier(
                model,
                train_loader,
                validation_loader,
                device,
                int(config["target_epochs"]),
                encoder_lr,
                head_lr,
                float(config["weight_decay"]),
                2,
            )
            target_evaluation = evaluate_classifier(
                model,
                target_test_loader,
                device,
                2,
            )
            target_head_source_evaluation = evaluate_classifier(
                model,
                source_test_loader,
                device,
                2,
            )
            source_head_evaluator = create_source_head_evaluator(
                model,
                pretrained_model,
            )
            source_retention_evaluation = evaluate_classifier(
                source_head_evaluator,
                source_test_loader,
                device,
                2,
            )
            current_trial[method_name] = {
                "best_epoch": training.best_epoch,
                "target_test_accuracy": target_evaluation.accuracy,
                "target_test_loss": target_evaluation.loss,
                "source_accuracy_with_target_head": (
                    target_head_source_evaluation.accuracy
                ),
                "source_accuracy_with_original_source_head": (
                    source_retention_evaluation.accuracy
                ),
            }
            if trial_index == 0:
                first_trial_training[method_name] = training
            logger.info(
                "trial %d %s：target %.2f%%，source %.2f%%",
                trial_index + 1,
                method_name,
                target_evaluation.accuracy * 100.0,
                source_retention_evaluation.accuracy * 100.0,
            )
        trial_results.append(current_trial)

    target_aggregate = aggregate_trial_values(
        trial_results,
        "target_test_accuracy",
    )
    source_aggregate = aggregate_trial_values(
        trial_results,
        "source_accuracy_with_original_source_head",
    )
    save_first_trial_curves(first_trial_training)
    save_accuracy_comparison(
        target_aggregate,
        source_aggregate,
        pretrained_source_evaluation.accuracy,
    )
    results = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "device": str(device),
        "source_pretraining": {
            "best_epoch": source_training.best_epoch,
            "source_test_accuracy": pretrained_source_evaluation.accuracy,
            "rotated_target_zero_shot_accuracy": zero_shot_target.accuracy,
        },
        "target_labeled_sample_count_per_trial": int(
            config["target_train_sample_count"]
        ),
        "trial_results": trial_results,
        "target_test_accuracy_aggregate": target_aggregate,
        "source_accuracy_with_original_source_head_aggregate": (
            source_aggregate
        ),
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    logger.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
