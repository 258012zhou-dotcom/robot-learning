"""Train and evaluate a small single-object shape detector."""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from robot_learning.dynamics_model import select_torch_device
from robot_learning.model_artifacts import (
    image_preprocessing,
    run_inference_cli,
    save_model_artifact,
)
from robot_learning.object_detection import (
    SmallShapeDetector,
    batch_intersection_over_union,
    create_detection_data_loader,
    evaluate_detector,
    generate_shape_detection_dataset,
    normalized_cxcywh_to_xyxy,
    split_detection_dataset,
    train_detector,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "008_single_object_detection.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "008_single_object_detection"
CLASS_NAMES = ("square", "circle")


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


def save_training_curve(
    training_losses: list[float],
    validation_losses: list[float],
    best_epoch: int,
) -> None:
    """Save total training and validation loss curves."""
    epochs = np.arange(1, len(training_losses) + 1)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(epochs, training_losses, label="training total loss")
    axis.plot(epochs, validation_losses, label="validation total loss")
    axis.axvline(
        best_epoch,
        color="gray",
        linestyle="--",
        label="best validation epoch",
    )
    axis.set(
        title="Single-object detection training",
        xlabel="epoch",
        ylabel="weighted total loss",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curve.png", dpi=150)
    plt.close(figure)


def _draw_normalized_box(
    axis,
    box_cxcywh: torch.Tensor,
    image_size: int,
    color: str,
    label: str,
) -> None:
    """Draw one normalized center-format box on a Matplotlib axis."""
    xyxy = normalized_cxcywh_to_xyxy(box_cxcywh.unsqueeze(0))[0]
    x_min, y_min, x_max, y_max = (
        xyxy * image_size
    ).tolist()
    axis.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor=color,
            linewidth=2,
            label=label,
        )
    )


def save_detection_examples(
    images: torch.Tensor,
    target_labels: torch.Tensor,
    target_boxes: torch.Tensor,
    predicted_labels: torch.Tensor,
    predicted_boxes: torch.Tensor,
    ious: torch.Tensor,
) -> None:
    """Save twelve examples with target and predicted boxes."""
    image_size = images.shape[-1]
    figure, axes = plt.subplots(3, 4, figsize=(10, 8))
    for axis, image, target_label, target_box, predicted_label, predicted_box, iou in zip(
        axes.flat,
        images[:12],
        target_labels[:12],
        target_boxes[:12],
        predicted_labels[:12],
        predicted_boxes[:12],
        ious[:12],
    ):
        axis.imshow(image.permute(1, 2, 0).numpy())
        _draw_normalized_box(
            axis,
            target_box,
            image_size,
            color="lime",
            label="target",
        )
        _draw_normalized_box(
            axis,
            predicted_box,
            image_size,
            color="magenta",
            label="prediction",
        )
        axis.set_title(
            f"true={CLASS_NAMES[int(target_label)]} "
            f"pred={CLASS_NAMES[int(predicted_label)]}\n"
            f"IoU={float(iou):.2f}"
        )
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle("Green: target; magenta: prediction")
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(OUTPUT_DIR / "detection_examples.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Generate data, train the detector, evaluate, and save evidence."""
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

    images, labels, boxes = generate_shape_detection_dataset(
        sample_count=int(config["sample_count"]),
        image_size=int(config["image_size"]),
        seed=seed,
    )
    splits = split_detection_dataset(
        images,
        labels,
        boxes,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
        seed=seed + 1,
    )
    batch_size = int(config["batch_size"])
    train_loader = create_detection_data_loader(
        splits.train_images,
        splits.train_labels,
        splits.train_boxes,
        batch_size=batch_size,
        shuffle=True,
        seed=seed + 2,
    )
    validation_loader = create_detection_data_loader(
        splits.validation_images,
        splits.validation_labels,
        splits.validation_boxes,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 2,
    )
    test_loader = create_detection_data_loader(
        splits.test_images,
        splits.test_labels,
        splits.test_boxes,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 2,
    )

    device = select_torch_device()
    set_random_seeds(seed + 10)
    model = SmallShapeDetector(num_classes=len(CLASS_NAMES))
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    training = train_detector(
        model,
        train_loader,
        validation_loader,
        device=device,
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        box_loss_weight=float(config["box_loss_weight"]),
    )
    test_evaluation = evaluate_detector(
        model,
        test_loader,
        device=device,
        box_loss_weight=float(config["box_loss_weight"]),
        iou_success_threshold=float(config["iou_success_threshold"]),
    )

    majority_class = int(torch.mode(splits.train_labels).values)
    mean_train_box = splits.train_boxes.mean(dim=0, keepdim=True)
    baseline_labels = torch.full_like(
        splits.test_labels,
        fill_value=majority_class,
    )
    baseline_boxes = mean_train_box.repeat(splits.test_boxes.shape[0], 1)
    baseline_ious = batch_intersection_over_union(
        baseline_boxes,
        splits.test_boxes,
    )
    baseline_correct_classes = baseline_labels == splits.test_labels
    baseline_successes = baseline_correct_classes & (
        baseline_ious >= float(config["iou_success_threshold"])
    )
    baseline_class_accuracy = float(
        baseline_correct_classes.float().mean().item()
    )
    baseline_mean_iou = float(baseline_ious.mean().item())
    baseline_success_rate = float(
        baseline_successes.float().mean().item()
    )

    save_training_curve(
        training.training_losses,
        training.validation_losses,
        training.best_epoch,
    )
    save_detection_examples(
        splits.test_images,
        test_evaluation.target_labels,
        test_evaluation.target_boxes,
        test_evaluation.predicted_labels,
        test_evaluation.predicted_boxes,
        test_evaluation.ious,
    )

    # train_* restores the minimum-validation-loss state before returning.
    save_model_artifact(OUTPUT_DIR / "best_model.pt", model, {
        "model_kind": "shape_detector",
        "architecture": {"num_classes": len(CLASS_NAMES)},
        "class_names": list(CLASS_NAMES),
        "preprocessing": image_preprocessing(int(config["image_size"])),
        "training_config": config,
        "best_epoch": training.best_epoch,
        "best_validation_loss": training.best_validation_loss,
        "box_format": "normalized_cxcywh",
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
        "best_epoch": training.best_epoch,
        "best_validation_loss": training.best_validation_loss,
        "iou_success_threshold": float(config["iou_success_threshold"]),
        "baseline": {
            "class_accuracy": baseline_class_accuracy,
            "mean_iou": baseline_mean_iou,
            "detection_success_rate": baseline_success_rate,
        },
        "model_test": {
            "total_loss": test_evaluation.total_loss,
            "classification_loss": test_evaluation.classification_loss,
            "box_regression_loss": test_evaluation.box_regression_loss,
            "class_accuracy": test_evaluation.class_accuracy,
            "mean_iou": test_evaluation.mean_iou,
            "detection_success_rate": (
                test_evaluation.detection_success_rate
            ),
        },
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    logger.info("设备：%s", device)
    logger.info("模型参数量：%s", parameter_count)
    logger.info("最佳验证轮次：%s", training.best_epoch)
    logger.info(
        "基线：分类 %.2f%%，平均 IoU %.4f，检测成功率 %.2f%%",
        baseline_class_accuracy * 100.0,
        baseline_mean_iou,
        baseline_success_rate * 100.0,
    )
    logger.info(
        "模型：分类 %.2f%%，平均 IoU %.4f，检测成功率 %.2f%%",
        test_evaluation.class_accuracy * 100.0,
        test_evaluation.mean_iou,
        test_evaluation.detection_success_rate * 100.0,
    )
    logger.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_inference_cli("shape_detector")
    else:
        main()
