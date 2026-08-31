"""Train and evaluate a small CLIP-style image-text dual encoder."""

from copy import deepcopy
import json
import logging
from pathlib import Path
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW

from robot_learning.dynamics_model import select_torch_device
from robot_learning.vision_language import (
    ImageTextDataset,
    SemanticRetrievalEvaluation,
    SimpleVocabulary,
    VisionLanguageDualEncoder,
    calculate_image_text_similarities,
    evaluate_semantic_image_text_retrieval,
    generate_image_text_dataset,
    train_vision_language_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "014_vision_language_alignment.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "014_vision_language_alignment"


def load_config() -> dict[str, Any]:
    """Load experiment settings from JSON."""
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


def create_model(
    config: dict[str, Any],
    vocabulary: SimpleVocabulary,
) -> VisionLanguageDualEncoder:
    """Construct the shared-space image and text encoders."""
    return VisionLanguageDualEncoder(
        vocabulary_size=len(vocabulary),
        maximum_token_count=int(config["maximum_token_count"]),
        embedding_dimension=int(config["embedding_dimension"]),
        text_head_count=int(config["text_head_count"]),
        text_mlp_hidden_dimension=int(config["text_mlp_hidden_dimension"]),
        text_layer_count=int(config["text_layer_count"]),
        padding_id=vocabulary.padding_id,
    )


def evaluate(
    model: VisionLanguageDualEncoder,
    dataset: ImageTextDataset,
    concept_token_ids: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> SemanticRetrievalEvaluation:
    """Evaluate canonical semantic texts against all images."""
    return evaluate_semantic_image_text_retrieval(
        model,
        dataset.images,
        dataset.concept_labels,
        concept_token_ids,
        device,
        temperature=float(config["temperature"]),
        image_batch_size=int(config["evaluation_batch_size"]),
    )


def select_representative_indices(dataset: ImageTextDataset) -> torch.Tensor:
    """Select the first test image belonging to every semantic concept."""
    return torch.tensor(
        [
            int(torch.nonzero(dataset.concept_labels == label)[0].item())
            for label in range(len(dataset.concept_descriptions))
        ],
        dtype=torch.int64,
    )


def representative_similarities(
    model: VisionLanguageDualEncoder,
    dataset: ImageTextDataset,
    concept_token_ids: torch.Tensor,
    representative_indices: torch.Tensor,
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    """Return one image per concept against all canonical descriptions."""
    model.eval()
    with torch.no_grad():
        image_embeddings = model.image_encoder(
            dataset.images[representative_indices].to(device)
        )
        text_embeddings = model.text_encoder(concept_token_ids.to(device))
        return calculate_image_text_similarities(
            image_embeddings,
            text_embeddings,
            temperature,
        ).cpu()


def save_learning_curves(
    training_losses: list[float],
    validation_losses: list[float],
    validation_accuracies: list[float],
    best_epoch: int,
) -> None:
    """Save training loss and semantic validation evidence."""
    epochs = np.arange(1, len(training_losses) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, training_losses, label="training contrastive")
    axes[0].plot(epochs, validation_losses, label="validation classification")
    axes[0].axvline(best_epoch, color="black", linestyle="--", alpha=0.6)
    axes[0].set(title="learning losses", xlabel="epoch", ylabel="loss")
    axes[1].plot(epochs, validation_accuracies, color="tab:green")
    axes[1].axvline(best_epoch, color="black", linestyle="--", alpha=0.6)
    axes[1].set(
        title="validation image-to-text accuracy",
        xlabel="epoch",
        ylabel="accuracy",
        ylim=(0.0, 1.05),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend() if axis.get_legend_handles_labels()[0] else None
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150)
    plt.close(figure)


def save_similarity_matrices(
    baseline: torch.Tensor,
    trained: torch.Tensor,
    descriptions: tuple[str, ...],
) -> None:
    """Compare representative image-text similarities before and after training."""
    minimum = float(torch.minimum(baseline.min(), trained.min()).item())
    maximum = float(torch.maximum(baseline.max(), trained.max()).item())
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        layout="constrained",
    )
    for axis, matrix, title in (
        (axes[0], baseline, "untrained"),
        (axes[1], trained, "trained"),
    ):
        image = axis.imshow(matrix.numpy(), vmin=minimum, vmax=maximum)
        axis.set(
            title=title,
            xlabel="text concept",
            ylabel="image concept",
            xticks=np.arange(len(descriptions)),
            yticks=np.arange(len(descriptions)),
            xticklabels=descriptions,
            yticklabels=descriptions,
        )
        axis.tick_params(axis="x", rotation=60)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        pad=0.02,
        label="scaled cosine similarity",
    )
    figure.savefig(OUTPUT_DIR / "similarity_matrices.png", dpi=150)
    plt.close(figure)


def save_confusion_comparison(
    baseline: SemanticRetrievalEvaluation,
    trained: SemanticRetrievalEvaluation,
    descriptions: tuple[str, ...],
) -> None:
    """Save row-true image-to-text confusion matrices."""
    maximum = int(
        torch.maximum(
            baseline.confusion_matrix.max(),
            trained.confusion_matrix.max(),
        ).item()
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        layout="constrained",
    )
    for axis, matrix, title in (
        (axes[0], baseline.confusion_matrix, "untrained"),
        (axes[1], trained.confusion_matrix, "trained"),
    ):
        image = axis.imshow(matrix.numpy(), vmin=0, vmax=maximum, cmap="Blues")
        axis.set(
            title=title,
            xlabel="predicted text",
            ylabel="true image concept",
            xticks=np.arange(len(descriptions)),
            yticks=np.arange(len(descriptions)),
            xticklabels=descriptions,
            yticklabels=descriptions,
        )
        axis.tick_params(axis="x", rotation=60)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    str(int(matrix[row, column].item())),
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        pad=0.02,
        label="image count",
    )
    figure.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close(figure)


def metric_dictionary(
    evaluation: SemanticRetrievalEvaluation,
) -> dict[str, Any]:
    """Convert scalar metrics and confusion evidence to JSON values."""
    return {
        "loss": evaluation.loss,
        "image_to_text_accuracy": evaluation.image_to_text_accuracy,
        "text_to_image_accuracy": evaluation.text_to_image_accuracy,
        "mean_similarity_margin": evaluation.mean_similarity_margin,
        "confusion_matrix": evaluation.confusion_matrix.tolist(),
    }


def main() -> None:
    """Train the dual encoder and save reproducible alignment evidence."""
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

    dataset_arguments = {"image_size": int(config["image_size"])}
    train_dataset = generate_image_text_dataset(
        int(config["train_samples_per_concept"]),
        seed=seed,
        **dataset_arguments,
    )
    validation_dataset = generate_image_text_dataset(
        int(config["validation_samples_per_concept"]),
        seed=seed + 1,
        **dataset_arguments,
    )
    test_dataset = generate_image_text_dataset(
        int(config["test_samples_per_concept"]),
        seed=seed + 2,
        **dataset_arguments,
    )
    vocabulary = SimpleVocabulary.from_texts(
        train_dataset.concept_descriptions
    )
    maximum_token_count = int(config["maximum_token_count"])
    train_token_ids = vocabulary.encode_batch(
        train_dataset.descriptions,
        maximum_token_count,
    )
    concept_token_ids = vocabulary.encode_batch(
        train_dataset.concept_descriptions,
        maximum_token_count,
    )
    model = create_model(config, vocabulary).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    baseline_evaluation = evaluate(
        model,
        test_dataset,
        concept_token_ids,
        config,
        device,
    )
    representative_indices = select_representative_indices(test_dataset)
    baseline_matrix = representative_similarities(
        model,
        test_dataset,
        concept_token_ids,
        representative_indices,
        device,
        float(config["temperature"]),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    training_losses: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(config["epochs"]) + 1):
        training_loss = train_vision_language_epoch(
            model,
            train_dataset.images,
            train_token_ids,
            train_dataset.concept_labels,
            optimizer,
            device,
            float(config["temperature"]),
            seed + epoch,
        )
        validation = evaluate(
            model,
            validation_dataset,
            concept_token_ids,
            config,
            device,
        )
        training_losses.append(training_loss)
        validation_losses.append(validation.loss)
        validation_accuracies.append(validation.image_to_text_accuracy)
        if validation.loss < best_validation_loss:
            best_validation_loss = validation.loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0:
            logger.info(
                "轮次 %d/%d：train %.4f，validation %.4f，accuracy %.2f%%",
                epoch,
                int(config["epochs"]),
                training_loss,
                validation.loss,
                validation.image_to_text_accuracy * 100.0,
            )
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    torch.save(best_state, OUTPUT_DIR / "best_model.pt")
    trained_evaluation = evaluate(
        model,
        test_dataset,
        concept_token_ids,
        config,
        device,
    )
    trained_matrix = representative_similarities(
        model,
        test_dataset,
        concept_token_ids,
        representative_indices,
        device,
        float(config["temperature"]),
    )

    save_learning_curves(
        training_losses,
        validation_losses,
        validation_accuracies,
        best_epoch,
    )
    save_similarity_matrices(
        baseline_matrix,
        trained_matrix,
        test_dataset.concept_descriptions,
    )
    save_confusion_comparison(
        baseline_evaluation,
        trained_evaluation,
        test_dataset.concept_descriptions,
    )
    results = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "device": str(device),
        "parameter_count": parameter_count,
        "concept_descriptions": list(test_dataset.concept_descriptions),
        "sample_counts": {
            "train": int(train_dataset.images.shape[0]),
            "validation": int(validation_dataset.images.shape[0]),
            "test": int(test_dataset.images.shape[0]),
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "final_training_loss": training_losses[-1],
        "untrained_baseline": metric_dictionary(baseline_evaluation),
        "trained_model": metric_dictionary(trained_evaluation),
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    logger.info("设备：%s", device)
    logger.info("模型参数量：%d", parameter_count)
    logger.info("最佳验证轮次：%d", best_epoch)
    logger.info(
        "未训练：I2T %.2f%%，T2I %.2f%%，margin %.4f",
        baseline_evaluation.image_to_text_accuracy * 100.0,
        baseline_evaluation.text_to_image_accuracy * 100.0,
        baseline_evaluation.mean_similarity_margin,
    )
    logger.info(
        "训练后：I2T %.2f%%，T2I %.2f%%，margin %.4f",
        trained_evaluation.image_to_text_accuracy * 100.0,
        trained_evaluation.text_to_image_accuracy * 100.0,
        trained_evaluation.mean_similarity_margin,
    )
    logger.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
