"""Compare untrained, supervised, and contrastive visual embeddings."""

from copy import deepcopy
import json
import logging
from pathlib import Path
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW

from robot_learning.dynamics_model import select_torch_device
from robot_learning.image_classification import (
    create_classification_data_loader,
    generate_shape_classification_dataset,
    split_classification_dataset,
    train_classifier,
)
from robot_learning.visual_representation import (
    EncoderClassifier,
    SmallVisualEncoder,
    calculate_embedding_statistics,
    create_contrastive_views,
    encode_images,
    evaluate_one_nearest_neighbor,
    train_contrastive_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "011_visual_representation.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "011_visual_representation"
CLASS_NAMES = ("square", "circle")
METHOD_NAMES = ("untrained", "supervised", "contrastive")


def load_config() -> dict[str, Any]:
    """Load the experiment JSON configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def embedding_method_results(
    encoder: SmallVisualEncoder,
    reference_images: Tensor,
    reference_labels: Tensor,
    test_images: Tensor,
    test_labels: Tensor,
    challenge_images: Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], Tensor]:
    """Evaluate one encoder with a shared frozen 1-NN protocol."""
    reference_embeddings = encode_images(
        encoder,
        reference_images,
        device,
        batch_size,
    )
    test_embeddings = encode_images(
        encoder,
        test_images,
        device,
        batch_size,
    )
    challenge_embeddings = encode_images(
        encoder,
        challenge_images,
        device,
        batch_size,
    )
    clean_evaluation = evaluate_one_nearest_neighbor(
        reference_embeddings,
        reference_labels,
        test_embeddings,
        test_labels,
    )
    challenge_evaluation = evaluate_one_nearest_neighbor(
        reference_embeddings,
        reference_labels,
        challenge_embeddings,
        test_labels,
    )
    statistics = calculate_embedding_statistics(test_embeddings)
    appearance_invariance = float(
        torch.nn.functional.cosine_similarity(
            test_embeddings,
            challenge_embeddings,
            dim=1,
        ).mean().item()
    )
    return (
        {
            "clean_test_1nn_accuracy": clean_evaluation.accuracy,
            "appearance_challenge_1nn_accuracy": challenge_evaluation.accuracy,
            "clean_challenge_mean_cosine_similarity": appearance_invariance,
            "mean_dimension_standard_deviation": (
                statistics.mean_dimension_standard_deviation
            ),
            "mean_pairwise_cosine_similarity": (
                statistics.mean_pairwise_cosine_similarity
            ),
        },
        test_embeddings,
    )


def project_embeddings_to_two_dimensions(embeddings: Tensor) -> np.ndarray:
    """Project embeddings with a centered two-component SVD for plotting."""
    values = embeddings.numpy()
    centered = values - values.mean(axis=0, keepdims=True)
    left_vectors, singular_values, _ = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    return left_vectors[:, :2] * singular_values[:2]


def save_embedding_plot(
    embeddings_by_method: dict[str, Tensor],
    labels: Tensor,
) -> None:
    """Save the same test labels in each method's two-dimensional projection."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    label_values = labels.numpy()
    for axis, method_name in zip(axes, METHOD_NAMES):
        projection = project_embeddings_to_two_dimensions(
            embeddings_by_method[method_name]
        )
        for class_index, class_name in enumerate(CLASS_NAMES):
            selected = label_values == class_index
            axis.scatter(
                projection[selected, 0],
                projection[selected, 1],
                s=14,
                alpha=0.65,
                label=class_name,
            )
        axis.set(
            title=method_name,
            xlabel="component 1",
            ylabel="component 2",
        )
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Frozen visual embeddings on the clean test split")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "embedding_comparison.png", dpi=150)
    plt.close(figure)


def save_accuracy_plot(results: dict[str, dict[str, float]]) -> None:
    """Save clean and appearance-shift 1-NN accuracy bars."""
    positions = np.arange(len(METHOD_NAMES))
    width = 0.36
    clean = [results[name]["clean_test_1nn_accuracy"] for name in METHOD_NAMES]
    challenge = [
        results[name]["appearance_challenge_1nn_accuracy"]
        for name in METHOD_NAMES
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(positions - width / 2, clean, width, label="clean test")
    axis.bar(
        positions + width / 2,
        challenge,
        width,
        label="appearance challenge",
    )
    axis.axhline(0.5, color="gray", linestyle="--", label="chance level")
    axis.set(
        title="Frozen-embedding 1-NN evaluation",
        ylabel="accuracy",
        xticks=positions,
        xticklabels=METHOD_NAMES,
        ylim=(0.0, 1.05),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(figure)


def save_training_curves(
    supervised_losses: list[float],
    contrastive_losses: list[float],
) -> None:
    """Save separate loss curves because the loss scales are not comparable."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        np.arange(1, len(supervised_losses) + 1),
        supervised_losses,
    )
    axes[0].set(
        title="supervised cross-entropy",
        xlabel="epoch",
        ylabel="loss",
    )
    axes[1].plot(
        np.arange(1, len(contrastive_losses) + 1),
        contrastive_losses,
    )
    axes[1].set(
        title="contrastive NT-Xent",
        xlabel="epoch",
        ylabel="loss",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(figure)


def save_augmentation_preview(
    clean_images: Tensor,
    challenge_images: Tensor,
) -> None:
    """Save paired clean and appearance-shift samples."""
    figure, axes = plt.subplots(2, 6, figsize=(12, 4))
    for column in range(6):
        axes[0, column].imshow(clean_images[column].permute(1, 2, 0).numpy())
        axes[1, column].imshow(
            challenge_images[column].permute(1, 2, 0).numpy()
        )
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("clean")
    axes[1, 0].set_ylabel("challenge")
    figure.suptitle("Appearance challenge preserves shape geometry")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "augmentation_preview.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Train three encoders and compare their frozen visual embeddings."""
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

    images, labels = generate_shape_classification_dataset(
        sample_count=int(config["sample_count"]),
        image_size=int(config["image_size"]),
        seed=seed,
    )
    splits = split_classification_dataset(
        images,
        labels,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
        seed=seed + 1,
    )
    batch_size = int(config["batch_size"])
    train_loader = create_classification_data_loader(
        splits.train_images,
        splits.train_labels,
        batch_size=batch_size,
        shuffle=True,
        seed=seed + 2,
    )
    validation_loader = create_classification_data_loader(
        splits.validation_images,
        splits.validation_labels,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 2,
    )
    device = select_torch_device()

    set_random_seeds(seed + 10)
    initial_encoder = SmallVisualEncoder(
        embedding_dimension=int(config["embedding_dimension"])
    )
    untrained_encoder = deepcopy(initial_encoder).to(device)
    supervised_encoder = deepcopy(initial_encoder)
    contrastive_encoder = deepcopy(initial_encoder).to(device)

    supervised_model = EncoderClassifier(supervised_encoder, num_classes=2)
    supervised_training = train_classifier(
        supervised_model,
        train_loader,
        validation_loader,
        device=device,
        epochs=int(config["supervised_epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        num_classes=2,
    )

    contrastive_optimizer = AdamW(
        contrastive_encoder.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    augmentation = config["training_augmentation"]
    contrastive_losses: list[float] = []
    for epoch_index in range(int(config["contrastive_epochs"])):
        loss = train_contrastive_epoch(
            contrastive_encoder,
            splits.train_images,
            contrastive_optimizer,
            device=device,
            batch_size=batch_size,
            temperature=float(config["temperature"]),
            seed=seed + 1000 + epoch_index,
            minimum_color_gain=float(augmentation["minimum_color_gain"]),
            maximum_color_gain=float(augmentation["maximum_color_gain"]),
            noise_standard_deviation=float(
                augmentation["noise_standard_deviation"]
            ),
        )
        contrastive_losses.append(loss)
        if (epoch_index + 1) % 10 == 0:
            logger.info(
                "对比学习轮次 %d/%d，NT-Xent：%.4f",
                epoch_index + 1,
                int(config["contrastive_epochs"]),
                loss,
            )

    challenge_config = config["appearance_challenge"]
    challenge_images, _ = create_contrastive_views(
        splits.test_images,
        seed=seed + 2000,
        minimum_color_gain=float(challenge_config["minimum_color_gain"]),
        maximum_color_gain=float(challenge_config["maximum_color_gain"]),
        noise_standard_deviation=float(
            challenge_config["noise_standard_deviation"]
        ),
    )
    encoders = {
        "untrained": untrained_encoder,
        "supervised": supervised_model.encoder,
        "contrastive": contrastive_encoder,
    }
    method_results: dict[str, dict[str, float]] = {}
    test_embeddings: dict[str, Tensor] = {}
    for method_name, encoder in encoders.items():
        result, embeddings = embedding_method_results(
            encoder,
            splits.train_images,
            splits.train_labels,
            splits.test_images,
            splits.test_labels,
            challenge_images,
            device,
            batch_size,
        )
        method_results[method_name] = result
        test_embeddings[method_name] = embeddings
        logger.info(
            "%s：clean 1-NN %.2f%%，appearance 1-NN %.2f%%，invariance %.4f",
            method_name,
            result["clean_test_1nn_accuracy"] * 100.0,
            result["appearance_challenge_1nn_accuracy"] * 100.0,
            result["clean_challenge_mean_cosine_similarity"],
        )

    save_training_curves(
        supervised_training.training_losses,
        contrastive_losses,
    )
    save_embedding_plot(test_embeddings, splits.test_labels)
    save_accuracy_plot(method_results)
    save_augmentation_preview(splits.test_images, challenge_images)

    results = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "device": str(device),
        "sample_counts": {
            "train": int(splits.train_images.shape[0]),
            "validation": int(splits.validation_images.shape[0]),
            "test": int(splits.test_images.shape[0]),
        },
        "embedding_dimension": int(config["embedding_dimension"]),
        "encoder_parameter_count": sum(
            parameter.numel() for parameter in initial_encoder.parameters()
        ),
        "supervised_training": {
            "best_epoch": supervised_training.best_epoch,
            "best_validation_loss": supervised_training.best_validation_loss,
            "final_training_loss": supervised_training.training_losses[-1],
        },
        "contrastive_training": {
            "final_nt_xent_loss": contrastive_losses[-1],
            "minimum_nt_xent_loss": min(contrastive_losses),
        },
        "methods": method_results,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    logger.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
