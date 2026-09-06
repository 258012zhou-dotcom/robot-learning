"""Train and evaluate a small synthetic semantic-segmentation model."""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_learning.dynamics_model import select_torch_device
from robot_learning.model_artifacts import (
    image_preprocessing,
    run_inference_cli,
    save_model_artifact,
)
from robot_learning.semantic_segmentation import (
    SmallSemanticSegmenter,
    calculate_segmentation_class_weights,
    create_segmentation_data_loader,
    evaluate_segmenter,
    generate_shape_segmentation_dataset,
    segmentation_confusion_matrix,
    segmentation_metrics_from_confusion,
    split_segmentation_dataset,
    train_segmenter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "009_semantic_segmentation.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "009_semantic_segmentation"
CLASS_NAMES = ("background", "square", "circle")
MASK_COLORS = np.array(
    [
        [0, 0, 0],
        [50, 220, 80],
        [255, 150, 40],
    ],
    dtype=np.uint8,
)


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


def colorize_mask(mask: torch.Tensor) -> np.ndarray:
    """Map integer class indices to RGB display colors."""
    return MASK_COLORS[mask.numpy()]


def save_training_curve(
    training_losses: list[float],
    validation_losses: list[float],
    best_epoch: int,
) -> None:
    """Save weighted pixel cross-entropy curves."""
    epochs = np.arange(1, len(training_losses) + 1)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(epochs, training_losses, label="training loss")
    axis.plot(epochs, validation_losses, label="validation loss")
    axis.axvline(
        best_epoch,
        color="gray",
        linestyle="--",
        label="best validation epoch",
    )
    axis.set(
        title="Semantic segmentation training",
        xlabel="epoch",
        ylabel="weighted pixel cross-entropy",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curve.png", dpi=150)
    plt.close(figure)


def save_confusion_matrix(confusion_matrix: torch.Tensor) -> None:
    """Save a row-normalized, row=true pixel confusion matrix."""
    matrix = confusion_matrix.to(dtype=torch.float64)
    normalized = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(normalized.numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(normalized.shape[0]):
        for column in range(normalized.shape[1]):
            axis.text(
                column,
                row,
                f"{normalized[row, column].item():.3f}",
                ha="center",
                va="center",
            )
    axis.set(
        title="Pixel confusion matrix (row normalized)",
        xlabel="predicted class",
        ylabel="true class",
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def save_segmentation_examples(
    images: torch.Tensor,
    target_masks: torch.Tensor,
    predicted_masks: torch.Tensor,
) -> None:
    """Save RGB, target, prediction, and prediction-overlay examples."""
    example_count = min(6, images.shape[0])
    figure, axes = plt.subplots(example_count, 4, figsize=(12, 3 * example_count))
    for row in range(example_count):
        rgb = images[row].permute(1, 2, 0).numpy()
        target_color = colorize_mask(target_masks[row])
        prediction_color = colorize_mask(predicted_masks[row])
        overlay = 0.65 * rgb + 0.35 * (prediction_color / 255.0)

        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(target_color)
        axes[row, 2].imshow(prediction_color)
        axes[row, 3].imshow(overlay.clip(0.0, 1.0))
        for column in range(4):
            axes[row, column].axis("off")
    for axis, title in zip(
        axes[0],
        ("RGB input", "target mask", "predicted mask", "prediction overlay"),
    ):
        axis.set_title(title)
    figure.suptitle("Background=black, square=green, circle=orange")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(OUTPUT_DIR / "segmentation_examples.png", dpi=150)
    plt.close(figure)


def metrics_to_dict(metrics) -> dict[str, Any]:
    """Convert segmentation metrics to a JSON-compatible dictionary."""
    return {
        "pixel_accuracy": metrics.pixel_accuracy,
        "per_class_iou": {
            class_name: float(class_iou)
            for class_name, class_iou in zip(
                CLASS_NAMES,
                metrics.per_class_iou.tolist(),
            )
        },
        "mean_iou": metrics.mean_iou,
        "foreground_mean_iou": metrics.foreground_mean_iou,
    }


def main() -> None:
    """Generate data, train the segmenter, evaluate, and save evidence."""
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

    images, masks = generate_shape_segmentation_dataset(
        sample_count=int(config["sample_count"]),
        image_size=int(config["image_size"]),
        seed=seed,
    )
    splits = split_segmentation_dataset(
        images,
        masks,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
        seed=seed + 1,
    )
    class_weights = calculate_segmentation_class_weights(
        splits.train_masks,
        num_classes=len(CLASS_NAMES),
    )
    batch_size = int(config["batch_size"])
    train_loader = create_segmentation_data_loader(
        splits.train_images,
        splits.train_masks,
        batch_size=batch_size,
        shuffle=True,
        seed=seed + 2,
    )
    validation_loader = create_segmentation_data_loader(
        splits.validation_images,
        splits.validation_masks,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 2,
    )
    test_loader = create_segmentation_data_loader(
        splits.test_images,
        splits.test_masks,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 2,
    )

    device = select_torch_device()
    set_random_seeds(seed + 10)
    model = SmallSemanticSegmenter(num_classes=len(CLASS_NAMES))
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    training = train_segmenter(
        model,
        train_loader,
        validation_loader,
        device=device,
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        class_weights=class_weights,
        num_classes=len(CLASS_NAMES),
    )
    evaluation = evaluate_segmenter(
        model,
        test_loader,
        device=device,
        class_weights=class_weights,
        num_classes=len(CLASS_NAMES),
    )

    baseline_predictions = torch.zeros_like(splits.test_masks)
    baseline_confusion = segmentation_confusion_matrix(
        baseline_predictions,
        splits.test_masks,
        num_classes=len(CLASS_NAMES),
    )
    baseline_metrics = segmentation_metrics_from_confusion(
        baseline_confusion
    )

    save_training_curve(
        training.training_losses,
        training.validation_losses,
        training.best_epoch,
    )
    save_confusion_matrix(evaluation.confusion_matrix)
    save_segmentation_examples(
        splits.test_images,
        evaluation.target_masks,
        evaluation.predicted_masks,
    )

    # train_* restores the minimum-validation-loss state before returning.
    save_model_artifact(OUTPUT_DIR / "best_model.pt", model, {
        "model_kind": "semantic_segmenter",
        "architecture": {"num_classes": len(CLASS_NAMES)},
        "class_names": list(CLASS_NAMES),
        "preprocessing": image_preprocessing(int(config["image_size"])),
        "training_config": config,
        "best_epoch": training.best_epoch,
        "best_validation_loss": training.best_validation_loss,
    })

    results = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "device": str(device),
        "parameter_count": parameter_count,
        "sample_counts": {
            "train": splits.train_images.shape[0],
            "validation": splits.validation_images.shape[0],
            "test": splits.test_images.shape[0],
        },
        "class_weights": {
            class_name: float(weight)
            for class_name, weight in zip(CLASS_NAMES, class_weights.tolist())
        },
        "best_epoch": training.best_epoch,
        "best_validation_loss": training.best_validation_loss,
        "all_background_baseline": metrics_to_dict(baseline_metrics),
        "model_test": {
            "loss": evaluation.loss,
            **metrics_to_dict(evaluation.metrics),
        },
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    logger.info("设备：%s", device)
    logger.info("模型参数量：%s", parameter_count)
    logger.info("最佳验证轮次：%s", training.best_epoch)
    logger.info(
        "全背景基线：Pixel Accuracy %.2f%%，mIoU %.4f，前景 mIoU %.4f",
        baseline_metrics.pixel_accuracy * 100.0,
        baseline_metrics.mean_iou,
        baseline_metrics.foreground_mean_iou,
    )
    logger.info(
        "模型：Pixel Accuracy %.2f%%，mIoU %.4f，前景 mIoU %.4f",
        evaluation.metrics.pixel_accuracy * 100.0,
        evaluation.metrics.mean_iou,
        evaluation.metrics.foreground_mean_iou,
    )
    for class_name, class_iou in zip(
        CLASS_NAMES,
        evaluation.metrics.per_class_iou.tolist(),
    ):
        logger.info("类别 %s IoU：%.4f", class_name, class_iou)
    logger.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_inference_cli("semantic_segmenter")
    else:
        main()
