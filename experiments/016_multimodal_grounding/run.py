"""Train and compare unimodal and multimodal visual-grounding baselines.

Recommended reading order:

1. Read ``main`` at the bottom for the complete experiment flow.
2. Read ``create_learned_model`` to compare model inputs.
3. Read the plotting functions only when studying result visualization.
"""

from dataclasses import asdict
import json
import logging
from pathlib import Path
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_learning.dynamics_model import select_torch_device
from robot_learning.multimodal_grounding import (
    GroundingDataset,
    LanguageOnlyGrounder,
    MultimodalGrounder,
    VisionOnlyGrounder,
    calculate_grounding_loss,
    generate_grounding_dataset,
    predict_fixed_center,
)
from robot_learning.multimodal_grounding_training import (
    GroundingEvaluation,
    GroundingModel,
    GroundingTrainingResult,
    evaluate_grounding_model,
    evaluate_grounding_predictions,
    train_grounding_model,
)
from robot_learning.vision_language import SimpleVocabulary


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "016_multimodal_grounding.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "016_multimodal_grounding"


# ---------------------------------------------------------------------------
# 1. Configuration, logging, and reproducibility
# ---------------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    """Read the experiment configuration from JSON."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def set_random_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before each model is created."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def configure_logging() -> logging.Logger:
    """Log concise progress to both the terminal and an output file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("multimodal_grounding_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "run.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# 2. Model construction and text preparation
# ---------------------------------------------------------------------------


def create_learned_model(
    method_name: str,
    config: dict[str, Any],
    vocabulary: SimpleVocabulary,
) -> GroundingModel:
    """Create a baseline with exactly the modalities named by the method."""
    common_text_arguments = {
        "vocabulary_size": len(vocabulary),
        "maximum_token_count": int(config["maximum_token_count"]),
        "embedding_dimension": int(config["embedding_dimension"]),
        "text_head_count": int(config["text_head_count"]),
        "padding_id": vocabulary.padding_id,
    }
    if method_name == "vision_only":
        return VisionOnlyGrounder(
            embedding_dimension=int(config["embedding_dimension"])
        )
    if method_name == "language_only":
        return LanguageOnlyGrounder(**common_text_arguments)
    if method_name == "multimodal":
        return MultimodalGrounder(**common_text_arguments)
    raise ValueError(f"unknown method: {method_name}")


def encode_dataset_instructions(
    vocabulary: SimpleVocabulary,
    dataset: GroundingDataset,
    maximum_token_count: int,
) -> torch.Tensor:
    """Convert human-readable commands into fixed-shape token IDs ``(N, L)``."""
    return vocabulary.encode_batch(
        dataset.instructions,
        maximum_token_count=maximum_token_count,
    )


# ---------------------------------------------------------------------------
# 3. Fixed baseline and shared learned-model experiment
# ---------------------------------------------------------------------------


def evaluate_fixed_center(dataset: GroundingDataset) -> GroundingEvaluation:
    """Evaluate the no-input baseline on the same test targets."""
    output = predict_fixed_center(dataset.images.shape[0])
    loss = calculate_grounding_loss(
        output.predicted_centers,
        dataset.target_centers,
    )
    return evaluate_grounding_predictions(
        predicted_centers=output.predicted_centers,
        target_centers=dataset.target_centers,
        scene_centers=dataset.scene_centers,
        target_indices=dataset.target_indices,
        loss=float(loss.item()),
    )


def train_and_evaluate_method(
    method_name: str,
    config: dict[str, Any],
    vocabulary: SimpleVocabulary,
    training_data: GroundingDataset,
    training_token_ids: torch.Tensor,
    validation_data: GroundingDataset,
    validation_token_ids: torch.Tensor,
    test_data: GroundingDataset,
    test_token_ids: torch.Tensor,
    device: torch.device,
    model_seed: int,
) -> tuple[GroundingModel, GroundingTrainingResult, GroundingEvaluation]:
    """Train one learned baseline, restore its best epoch, and test it once."""
    set_random_seeds(model_seed)
    model = create_learned_model(method_name, config, vocabulary)
    training_result = train_grounding_model(
        model=model,
        train_images=training_data.images,
        train_token_ids=training_token_ids,
        train_target_centers=training_data.target_centers,
        validation_images=validation_data.images,
        validation_token_ids=validation_token_ids,
        validation_target_centers=validation_data.target_centers,
        validation_scene_centers=validation_data.scene_centers,
        validation_target_indices=validation_data.target_indices,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        device=device,
        seed=model_seed,
    )
    evaluation = evaluate_grounding_model(
        model=model,
        images=test_data.images,
        token_ids=test_token_ids,
        target_centers=test_data.target_centers,
        scene_centers=test_data.scene_centers,
        target_indices=test_data.target_indices,
        batch_size=int(config["batch_size"]),
        device=device,
    )
    return model, training_result, evaluation


# ---------------------------------------------------------------------------
# 4. Result visualization
# ---------------------------------------------------------------------------


def save_learning_curves(
    training_results: dict[str, GroundingTrainingResult],
) -> None:
    """Compare optimization and validation localization across learned methods."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method_name, result in training_results.items():
        epochs = np.arange(1, len(result.training_losses) + 1)
        axes[0].plot(epochs, result.training_losses, label=method_name)
        axes[1].plot(
            epochs,
            result.validation_center_errors,
            label=method_name,
        )
    axes[0].set(
        title="Training coordinate loss",
        xlabel="epoch",
        ylabel="Smooth L1 loss",
    )
    axes[1].set(
        title="Validation center error",
        xlabel="epoch",
        ylabel="normalized Euclidean error",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "learning_curves.png", dpi=160)
    plt.close(figure)


@torch.no_grad()
def save_multimodal_predictions(
    model: MultimodalGrounder,
    dataset: GroundingDataset,
    token_ids: torch.Tensor,
    device: torch.device,
    sample_count: int = 6,
) -> None:
    """Overlay attention, target centers, and predictions on test images."""
    model.eval()
    output = model(
        dataset.images[:sample_count].to(device),
        token_ids[:sample_count].to(device),
    )
    predictions = output.predicted_centers.cpu()
    attention = output.attention_weights
    if attention is None:
        raise RuntimeError("multimodal model did not return spatial attention")
    attention = attention.cpu()
    grid_size = int(round(attention.shape[1] ** 0.5))

    figure, axes = plt.subplots(2, 3, figsize=(11, 7), squeeze=False)
    for sample_index, axis in enumerate(axes.ravel()):
        image = dataset.images[sample_index].permute(1, 2, 0).numpy()
        image_height, image_width = image.shape[:2]
        target = dataset.target_centers[sample_index]
        prediction = predictions[sample_index]

        axis.imshow(image)
        axis.imshow(
            attention[sample_index].reshape(grid_size, grid_size),
            cmap="magma",
            alpha=0.38,
            extent=(0, image_width, image_height, 0),
            interpolation="bilinear",
        )
        axis.scatter(
            float(target[0]) * image_width,
            float(target[1]) * image_height,
            marker="o",
            s=120,
            facecolors="none",
            edgecolors="lime",
            linewidths=2,
            label="target",
        )
        axis.scatter(
            float(prediction[0]) * image_width,
            float(prediction[1]) * image_height,
            marker="x",
            s=90,
            color="cyan",
            linewidths=2,
            label="prediction",
        )
        axis.set_title(dataset.instructions[sample_index], fontsize=10)
        axis.set_axis_off()
    axes[0, 0].legend(loc="lower left", fontsize=8)
    figure.suptitle("Multimodal grounding: attention and coordinates")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "multimodal_predictions.png", dpi=160)
    plt.close(figure)


# ---------------------------------------------------------------------------
# 5. JSON serialization
# ---------------------------------------------------------------------------


def serialize_evaluation(evaluation: GroundingEvaluation) -> dict[str, float]:
    """Keep scalar metrics while excluding the large prediction tensor."""
    return {
        "loss": evaluation.loss,
        "mean_center_error": evaluation.mean_center_error,
        "selection_accuracy": evaluation.selection_accuracy,
    }


def serialize_training(
    result: GroundingTrainingResult,
) -> dict[str, Any]:
    """Convert the training dataclass into JSON-compatible values."""
    return asdict(result)


# ---------------------------------------------------------------------------
# 6. Complete experiment flow
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate data, compare four methods, and save reproducible evidence."""
    config = load_config()
    logger = configure_logging()
    seed = int(config["seed"])
    set_random_seeds(seed)
    device = select_torch_device()

    # Different seeds create independent train, validation, and test scenes.
    data_arguments = {
        "image_size": int(config["image_size"]),
        "object_count": int(config["object_count"]),
    }
    training_data = generate_grounding_dataset(
        sample_count=int(config["train_sample_count"]),
        seed=seed,
        **data_arguments,
    )
    validation_data = generate_grounding_dataset(
        sample_count=int(config["validation_sample_count"]),
        seed=seed + 1,
        **data_arguments,
    )
    test_data = generate_grounding_dataset(
        sample_count=int(config["test_sample_count"]),
        seed=seed + 2,
        **data_arguments,
    )

    canonical_instructions = tuple(
        f"select the {description}"
        for description in training_data.concept_descriptions
    )
    vocabulary = SimpleVocabulary.from_texts(canonical_instructions)
    maximum_token_count = int(config["maximum_token_count"])
    training_token_ids = encode_dataset_instructions(
        vocabulary,
        training_data,
        maximum_token_count,
    )
    validation_token_ids = encode_dataset_instructions(
        vocabulary,
        validation_data,
        maximum_token_count,
    )
    test_token_ids = encode_dataset_instructions(
        vocabulary,
        test_data,
        maximum_token_count,
    )

    logger.info("设备：%s", device)
    logger.info(
        "数据量：train=%s validation=%s test=%s",
        training_data.images.shape[0],
        validation_data.images.shape[0],
        test_data.images.shape[0],
    )
    logger.info("每个场景物体数：%s", config["object_count"])

    evaluations = {"fixed_center": evaluate_fixed_center(test_data)}
    training_results: dict[str, GroundingTrainingResult] = {}
    models: dict[str, GroundingModel] = {}

    for method_offset, method_name in enumerate(
        ("vision_only", "language_only", "multimodal"),
        start=1,
    ):
        model, training_result, evaluation = train_and_evaluate_method(
            method_name=method_name,
            config=config,
            vocabulary=vocabulary,
            training_data=training_data,
            training_token_ids=training_token_ids,
            validation_data=validation_data,
            validation_token_ids=validation_token_ids,
            test_data=test_data,
            test_token_ids=test_token_ids,
            device=device,
            model_seed=seed + 100 * method_offset,
        )
        models[method_name] = model
        training_results[method_name] = training_result
        evaluations[method_name] = evaluation
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        logger.info(
            "%s：best epoch %s，参数 %s，选择准确率 %.2f%%，中心误差 %.4f",
            method_name,
            training_result.best_epoch,
            parameter_count,
            100.0 * evaluation.selection_accuracy,
            evaluation.mean_center_error,
        )

    save_learning_curves(training_results)
    multimodal_model = models["multimodal"]
    if not isinstance(multimodal_model, MultimodalGrounder):
        raise RuntimeError("multimodal method returned an unexpected model")
    save_multimodal_predictions(
        model=multimodal_model,
        dataset=test_data,
        token_ids=test_token_ids,
        device=device,
    )
    torch.save(
        multimodal_model.state_dict(),
        OUTPUT_DIR / "best_multimodal_model.pt",
    )

    results = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "device": str(device),
        "sample_counts": {
            "train": training_data.images.shape[0],
            "validation": validation_data.images.shape[0],
            "test": test_data.images.shape[0],
        },
        "object_count": int(config["object_count"]),
        "methods": {
            name: serialize_evaluation(evaluation)
            for name, evaluation in evaluations.items()
        },
        "training": {
            name: serialize_training(result)
            for name, result in training_results.items()
        },
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    logger.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
