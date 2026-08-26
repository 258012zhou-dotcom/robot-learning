"""Train and evaluate a small CNN on synthetic shape images."""

import json
import logging
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_learning.dynamics_model import select_torch_device
from robot_learning.image_classification import (
    SmallShapeCNN,
    create_classification_data_loader,
    evaluate_classifier,
    generate_shape_classification_dataset,
    generate_shape_challenge_dataset,
    generate_rotation_augmented_shape_dataset,
    split_classification_dataset,
    train_classifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "007_cnn_classification.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "007_cnn_classification"
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
    filename: str,
    title: str,
) -> None:
    """Save training and validation cross-entropy curves."""
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
        title=title,
        xlabel="epoch",
        ylabel="cross-entropy loss",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(figure)


def save_confusion_matrix(
    confusion_matrix: torch.Tensor,
    filename: str,
    title: str,
) -> None:
    """Save a row=true, column=predicted confusion matrix."""
    matrix = confusion_matrix.numpy()
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )
    axis.set(
        title=title,
        xlabel="predicted class",
        ylabel="true class",
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(figure)


def save_misclassified_examples(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    filename: str,
    title: str,
) -> int:
    """Save up to twelve test mistakes and return the total mistake count."""
    mistake_indices = torch.nonzero(
        labels != predictions,
        as_tuple=False,
    ).flatten()
    figure, axes = plt.subplots(3, 4, figsize=(8, 6))
    for axis in axes.flat:
        axis.axis("off")

    if mistake_indices.numel() == 0:
        axes.flat[0].text(
            0.5,
            0.5,
            "No test misclassifications",
            ha="center",
            va="center",
        )
    else:
        for axis, index in zip(axes.flat, mistake_indices[:12]):
            true_label = int(labels[index])
            predicted_label = int(predictions[index])
            axis.imshow(images[index].permute(1, 2, 0).numpy())
            axis.set_title(
                f"true={CLASS_NAMES[true_label]}\n"
                f"pred={CLASS_NAMES[predicted_label]}"
            )
            axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(figure)
    return int(mistake_indices.numel())


def save_image_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    filename: str,
    title: str,
) -> None:
    """Save up to twelve labeled RGB Tensor images."""
    figure, axes = plt.subplots(3, 4, figsize=(8, 6))
    for axis in axes.flat:
        axis.axis("off")
    for axis, image, label in zip(axes.flat, images[:12], labels[:12]):
        label_index = int(label)
        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_title(f"label={CLASS_NAMES[label_index]}")
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(figure)


def main() -> None:
    """Generate data, train the CNN, evaluate, and save evidence."""
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
    augmentation_config = config["rotation_augmentation"]
    augmented_train_images, augmented_train_labels = (
        generate_rotation_augmented_shape_dataset(
            sample_count=splits.train_images.shape[0],
            image_size=int(config["image_size"]),
            seed=seed + 3,
            probability=float(augmentation_config["probability"]),
            maximum_angle_degrees=float(
                augmentation_config["maximum_angle_degrees"]
            ),
        )
    )
    augmented_train_loader = create_classification_data_loader(
        augmented_train_images,
        augmented_train_labels,
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
    test_loader = create_classification_data_loader(
        splits.test_images,
        splits.test_labels,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 2,
    )

    device = select_torch_device()
    set_random_seeds(seed + 10)
    model = SmallShapeCNN(num_classes=len(CLASS_NAMES))
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    training = train_classifier(
        model,
        train_loader,
        validation_loader,
        device=device,
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        num_classes=len(CLASS_NAMES),
    )
    test_evaluation = evaluate_classifier(
        model,
        test_loader,
        device,
        num_classes=len(CLASS_NAMES),
    )

    challenge_images, challenge_labels = generate_shape_challenge_dataset(
        sample_count=int(config["challenge_sample_count"]),
        image_size=int(config["image_size"]),
        seed=seed + 100,
    )
    challenge_loader = create_classification_data_loader(
        challenge_images,
        challenge_labels,
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 2,
    )
    challenge_evaluation = evaluate_classifier(
        model,
        challenge_loader,
        device,
        num_classes=len(CLASS_NAMES),
    )

    set_random_seeds(seed + 10)
    augmented_model = SmallShapeCNN(num_classes=len(CLASS_NAMES))
    augmented_training = train_classifier(
        augmented_model,
        augmented_train_loader,
        validation_loader,
        device=device,
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        num_classes=len(CLASS_NAMES),
    )
    augmented_test_evaluation = evaluate_classifier(
        augmented_model,
        test_loader,
        device,
        num_classes=len(CLASS_NAMES),
    )
    augmented_challenge_evaluation = evaluate_classifier(
        augmented_model,
        challenge_loader,
        device,
        num_classes=len(CLASS_NAMES),
    )

    class_counts = torch.bincount(
        splits.test_labels,
        minlength=len(CLASS_NAMES),
    )
    majority_baseline_accuracy = float(
        class_counts.max().item() / splits.test_labels.numel()
    )
    mistake_count = save_misclassified_examples(
        splits.test_images,
        test_evaluation.labels,
        test_evaluation.predictions,
        filename="misclassified_examples.png",
        title="Misclassified IID Test Samples",
    )
    challenge_mistake_count = save_misclassified_examples(
        challenge_images,
        challenge_evaluation.labels,
        challenge_evaluation.predictions,
        filename="challenge_misclassified_examples.png",
        title="Misclassified OOD Challenge Samples",
    )
    augmented_challenge_mistake_count = save_misclassified_examples(
        challenge_images,
        augmented_challenge_evaluation.labels,
        augmented_challenge_evaluation.predictions,
        filename="augmented_challenge_misclassified_examples.png",
        title="Augmented Model OOD Mistakes",
    )
    save_training_curve(
        training.training_losses,
        training.validation_losses,
        training.best_epoch,
        filename="learning_curves.png",
        title="Baseline CNN Learning Curves",
    )
    save_training_curve(
        augmented_training.training_losses,
        augmented_training.validation_losses,
        augmented_training.best_epoch,
        filename="augmented_learning_curves.png",
        title="Rotation-Augmented CNN Learning Curves",
    )
    save_confusion_matrix(
        test_evaluation.confusion_matrix,
        filename="confusion_matrix.png",
        title="IID Test Confusion Matrix",
    )
    save_confusion_matrix(
        challenge_evaluation.confusion_matrix,
        filename="challenge_confusion_matrix.png",
        title="OOD Challenge Confusion Matrix",
    )
    save_confusion_matrix(
        augmented_challenge_evaluation.confusion_matrix,
        filename="augmented_challenge_confusion_matrix.png",
        title="Augmented Model OOD Confusion Matrix",
    )
    save_image_grid(
        challenge_images,
        challenge_labels,
        filename="challenge_preview.png",
        title="OOD Challenge Samples",
    )
    save_image_grid(
        augmented_train_images,
        augmented_train_labels,
        filename="augmentation_preview.png",
        title="Rotation-Randomized Training Samples",
    )

    checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "class_names": CLASS_NAMES,
        "config": config,
    }
    torch.save(checkpoint, OUTPUT_DIR / "best_model.pt")
    augmented_checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in augmented_model.state_dict().items()
        },
        "class_names": CLASS_NAMES,
        "config": config,
    }
    torch.save(
        augmented_checkpoint,
        OUTPUT_DIR / "best_augmented_model.pt",
    )

    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )
    results = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "device": str(device),
        "device_name": device_name,
        "parameter_count": parameter_count,
        "split_sizes": {
            "train": splits.train_images.shape[0],
            "validation": splits.validation_images.shape[0],
            "test": splits.test_images.shape[0],
        },
        "best_epoch": training.best_epoch,
        "best_validation_loss": training.best_validation_loss,
        "best_validation_accuracy": training.validation_accuracies[
            training.best_epoch - 1
        ],
        "test_loss": test_evaluation.loss,
        "test_accuracy": test_evaluation.accuracy,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "test_confusion_matrix": test_evaluation.confusion_matrix.tolist(),
        "test_mistake_count": mistake_count,
        "challenge_loss": challenge_evaluation.loss,
        "challenge_accuracy": challenge_evaluation.accuracy,
        "challenge_confusion_matrix": (
            challenge_evaluation.confusion_matrix.tolist()
        ),
        "challenge_mistake_count": challenge_mistake_count,
        "rotation_augmentation": augmentation_config,
        "augmented_best_epoch": augmented_training.best_epoch,
        "augmented_best_validation_loss": (
            augmented_training.best_validation_loss
        ),
        "augmented_test_accuracy": augmented_test_evaluation.accuracy,
        "augmented_challenge_accuracy": (
            augmented_challenge_evaluation.accuracy
        ),
        "augmented_challenge_confusion_matrix": (
            augmented_challenge_evaluation.confusion_matrix.tolist()
        ),
        "augmented_challenge_mistake_count": (
            augmented_challenge_mistake_count
        ),
        "challenge_accuracy_gain": (
            augmented_challenge_evaluation.accuracy
            - challenge_evaluation.accuracy
        ),
    }
    with (OUTPUT_DIR / "results.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    logger.info("设备：%s (%s)", device, device_name)
    logger.info("参数量：%s", parameter_count)
    logger.info("数据划分：%s", results["split_sizes"])
    logger.info("最佳轮次：%s", training.best_epoch)
    logger.info("测试准确率：%.4f", test_evaluation.accuracy)
    logger.info("多数类基线：%.4f", majority_baseline_accuracy)
    logger.info("错误样本数：%s", mistake_count)
    logger.info("OOD 挑战准确率：%.4f", challenge_evaluation.accuracy)
    logger.info("OOD 挑战错误数：%s", challenge_mistake_count)
    logger.info(
        "旋转增强模型测试准确率：%.4f",
        augmented_test_evaluation.accuracy,
    )
    logger.info(
        "旋转增强模型 OOD 准确率：%.4f",
        augmented_challenge_evaluation.accuracy,
    )
    logger.info(
        "OOD 准确率提升：%+.4f",
        results["challenge_accuracy_gain"],
    )
    logger.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
