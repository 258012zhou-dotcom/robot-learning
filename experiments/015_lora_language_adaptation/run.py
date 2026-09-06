"""Compare frozen, LoRA, and full fine-tuning for language aliases."""

from copy import deepcopy
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW

from robot_learning.dynamics_model import select_torch_device
from robot_learning.model_artifacts import (
    image_preprocessing,
    run_inference_cli,
    save_model_artifact,
)
from robot_learning.peft import inject_lora_into_linear_layers
from robot_learning.vision_language import (
    ImageTextDataset,
    SemanticRetrievalEvaluation,
    SimpleVocabulary,
    VisionLanguageDualEncoder,
    evaluate_semantic_image_text_retrieval,
    generate_image_text_dataset,
    train_vision_language_epoch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "015_lora_language_adaptation.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "015_lora_language_adaptation"
METHOD_NAMES = ("frozen", "lora", "full_finetune")
TARGET_DESCRIPTIONS = (
    "crimson box",
    "crimson disk",
    "emerald box",
    "emerald disk",
    "azure box",
    "azure disk",
)
LORA_TARGETS = (
    "text_encoder.transformer.layers.0.linear1",
    "text_encoder.transformer.layers.0.linear2",
)


def load_config() -> dict[str, Any]:
    """Load experiment settings from JSON."""
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


def target_descriptions_for_dataset(dataset: ImageTextDataset) -> tuple[str, ...]:
    """Return one alias description for each image concept label."""
    return tuple(
        TARGET_DESCRIPTIONS[label]
        for label in dataset.concept_labels.tolist()
    )


def create_model(
    config: dict[str, Any],
    vocabulary: SimpleVocabulary,
) -> VisionLanguageDualEncoder:
    """Create the shared base architecture."""
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
    """Evaluate one set of concept prompts against an image dataset."""
    return evaluate_semantic_image_text_retrieval(
        model,
        dataset.images,
        dataset.concept_labels,
        concept_token_ids,
        device,
        temperature=float(config["temperature"]),
        image_batch_size=int(config["evaluation_batch_size"]),
    )


def save_lora_checkpoint(
    path: Path,
    model: VisionLanguageDualEncoder,
    config: dict[str, Any],
    vocabulary: SimpleVocabulary,
    source_descriptions: tuple[str, ...],
    training: dict[str, Any],
) -> None:
    """Bundle the selected adapter and its exact frozen base in one artifact."""
    architecture = {
        name: int(config[name]) for name in (
            "maximum_token_count", "embedding_dimension", "text_head_count",
            "text_mlp_hidden_dimension", "text_layer_count",
        )
    }
    architecture.update(vocabulary_size=len(vocabulary), padding_id=vocabulary.padding_id)
    tokens = sorted(vocabulary.token_to_id, key=vocabulary.token_to_id.__getitem__)
    save_model_artifact(path, model, {
        "model_kind": "lora_dual_encoder",
        "architecture": architecture,
        "vocabulary_tokens": tokens,
        "tokenization": "lowercase [a-z0-9]+; right pad; reject overlength",
        "lora": {
            "target_module_names": list(LORA_TARGETS),
            "rank": int(config["lora_rank"]),
            "alpha": float(config["lora_alpha"]),
            "dropout_probability": float(config["lora_dropout_probability"]),
        },
        "preprocessing": image_preprocessing(int(config["image_size"])),
        "source_descriptions": list(source_descriptions),
        "target_descriptions": list(TARGET_DESCRIPTIONS),
        "training_config": config,
        "best_epoch": training["best_epoch"],
        "best_validation_loss": training["best_validation_loss"],
    })


def train_with_validation(
    model: VisionLanguageDualEncoder,
    train_dataset: ImageTextDataset,
    train_token_ids: torch.Tensor,
    validation_dataset: ImageTextDataset,
    validation_concept_token_ids: torch.Tensor,
    optimizer: AdamW,
    epochs: int,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Train, select the minimum validation loss, and restore that state."""
    training_losses: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
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
            validation_concept_token_ids,
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
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return {
        "training_losses": training_losses,
        "validation_losses": validation_losses,
        "validation_accuracies": validation_accuracies,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }


def metric_dictionary(
    evaluation: SemanticRetrievalEvaluation,
) -> dict[str, float]:
    """Keep scalar semantic retrieval metrics for JSON output."""
    return {
        "loss": evaluation.loss,
        "image_to_text_accuracy": evaluation.image_to_text_accuracy,
        "text_to_image_accuracy": evaluation.text_to_image_accuracy,
        "mean_similarity_margin": evaluation.mean_similarity_margin,
    }


def aggregate_trials(
    trials: list[dict[str, Any]],
    domain_name: str,
    metric_name: str,
) -> dict[str, dict[str, float]]:
    """Return mean and population standard deviation per method."""
    aggregate: dict[str, dict[str, float]] = {}
    for method_name in METHOD_NAMES:
        values = np.array(
            [trial[method_name][domain_name][metric_name] for trial in trials],
            dtype=np.float64,
        )
        aggregate[method_name] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=0)),
        }
    return aggregate


def save_accuracy_comparison(
    target_aggregate: dict[str, dict[str, float]],
    source_aggregate: dict[str, dict[str, float]],
) -> None:
    """Save alias adaptation and original-language retention accuracy."""
    positions = np.arange(len(METHOD_NAMES))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, aggregate, title in (
        (axes[0], target_aggregate, "target alias prompts"),
        (axes[1], source_aggregate, "source canonical prompts"),
    ):
        means = [aggregate[name]["mean"] for name in METHOD_NAMES]
        errors = [
            aggregate[name]["standard_deviation"] for name in METHOD_NAMES
        ]
        bars = axis.bar(positions, means, yerr=errors, capsize=5)
        axis.set(
            title=title,
            ylabel="image-to-text accuracy",
            ylim=(0.0, 1.05),
            xticks=positions,
            xticklabels=METHOD_NAMES,
        )
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value * 100:.1f}%",
                ha="center",
                fontsize=9,
            )
    figure.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(figure)


def save_parameter_comparison(trainable_counts: dict[str, int]) -> None:
    """Save trainable parameter counts on a logarithmic axis."""
    display_counts = [max(trainable_counts[name], 1) for name in METHOD_NAMES]
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    bars = axis.bar(METHOD_NAMES, display_counts)
    axis.set(
        title="trainable parameter count",
        ylabel="parameters (log scale)",
        yscale="log",
    )
    axis.grid(axis="y", alpha=0.25)
    for bar, name in zip(bars, METHOD_NAMES):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            display_counts[METHOD_NAMES.index(name)] * 1.15,
            str(trainable_counts[name]),
            ha="center",
        )
    figure.savefig(OUTPUT_DIR / "trainable_parameters.png", dpi=150)
    plt.close(figure)


def save_first_trial_curves(first_trial_training: dict[str, dict[str, Any]]) -> None:
    """Save LoRA and full fine-tuning validation curves from trial one."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for method_name, training in first_trial_training.items():
        epochs = np.arange(1, len(training["training_losses"]) + 1)
        axes[0].plot(epochs, training["validation_losses"], label=method_name)
        axes[1].plot(
            epochs,
            training["validation_accuracies"],
            label=method_name,
        )
    axes[0].set(title="target validation loss", xlabel="epoch", ylabel="loss")
    axes[1].set(
        title="target validation accuracy",
        xlabel="epoch",
        ylabel="accuracy",
        ylim=(0.0, 1.05),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(OUTPUT_DIR / "first_trial_learning_curves.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Pretrain once, run repeated alias adaptation, and compare methods."""
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
    image_size = int(config["image_size"])
    maximum_token_count = int(config["maximum_token_count"])

    source_train = generate_image_text_dataset(
        int(config["source_train_samples_per_concept"]), image_size, seed
    )
    source_validation = generate_image_text_dataset(
        int(config["source_validation_samples_per_concept"]),
        image_size,
        seed + 1,
    )
    source_test = generate_image_text_dataset(
        int(config["source_test_samples_per_concept"]),
        image_size,
        seed + 2,
    )
    target_validation = generate_image_text_dataset(
        int(config["target_validation_samples_per_concept"]),
        image_size,
        seed + 3,
    )
    target_test = generate_image_text_dataset(
        int(config["target_test_samples_per_concept"]),
        image_size,
        seed + 4,
    )
    source_concepts = source_train.concept_descriptions
    vocabulary = SimpleVocabulary.from_texts(source_concepts + TARGET_DESCRIPTIONS)
    source_train_token_ids = vocabulary.encode_batch(
        source_train.descriptions,
        maximum_token_count,
    )
    source_concept_token_ids = vocabulary.encode_batch(
        source_concepts,
        maximum_token_count,
    )
    target_concept_token_ids = vocabulary.encode_batch(
        TARGET_DESCRIPTIONS,
        maximum_token_count,
    )

    set_random_seeds(seed + 10)
    base_model = create_model(config, vocabulary).to(device)
    base_optimizer = AdamW(
        base_model.parameters(),
        lr=float(config["source_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    source_training = train_with_validation(
        base_model,
        source_train,
        source_train_token_ids,
        source_validation,
        source_concept_token_ids,
        base_optimizer,
        int(config["source_epochs"]),
        config,
        device,
        seed + 20,
    )
    base_source = evaluate(
        base_model,
        source_test,
        source_concept_token_ids,
        config,
        device,
    )
    base_target = evaluate(
        base_model,
        target_test,
        target_concept_token_ids,
        config,
        device,
    )
    base_parameter_count = sum(
        parameter.numel() for parameter in base_model.parameters()
    )
    trainable_counts = {
        "frozen": 0,
        "lora": 0,
        "full_finetune": base_parameter_count,
    }

    trials: list[dict[str, Any]] = []
    first_trial_training: dict[str, dict[str, Any]] = {}
    for trial_index in range(int(config["trial_count"])):
        trial_seed = seed + 100 + trial_index
        target_train = generate_image_text_dataset(
            int(config["target_train_samples_per_concept"]),
            image_size,
            trial_seed,
        )
        target_train_token_ids = vocabulary.encode_batch(
            target_descriptions_for_dataset(target_train),
            maximum_token_count,
        )
        trial_result: dict[str, Any] = {
            "frozen": {
                "target": metric_dictionary(base_target),
                "source": metric_dictionary(base_source),
                "best_epoch": 0,
            }
        }

        lora_model = deepcopy(base_model)
        lora_report = inject_lora_into_linear_layers(
            lora_model,
            LORA_TARGETS,
            rank=int(config["lora_rank"]),
            alpha=float(config["lora_alpha"]),
            dropout_probability=float(config["lora_dropout_probability"]),
        )
        trainable_counts["lora"] = lora_report.trainable_parameter_count
        lora_optimizer = AdamW(
            [
                parameter
                for parameter in lora_model.parameters()
                if parameter.requires_grad
            ],
            lr=float(config["lora_learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        lora_training = train_with_validation(
            lora_model,
            target_train,
            target_train_token_ids,
            target_validation,
            target_concept_token_ids,
            lora_optimizer,
            int(config["target_epochs"]),
            config,
            device,
            trial_seed,
        )
        lora_target = evaluate(
            lora_model,
            target_test,
            target_concept_token_ids,
            config,
            device,
        )
        lora_source = evaluate(
            lora_model,
            source_test,
            source_concept_token_ids,
            config,
            device,
        )
        trial_result["lora"] = {
            "target": metric_dictionary(lora_target),
            "source": metric_dictionary(lora_source),
            "best_epoch": lora_training["best_epoch"],
            "best_validation_loss": lora_training["best_validation_loss"],
        }

        full_model = deepcopy(base_model)
        for parameter in full_model.parameters():
            parameter.requires_grad = True
        full_optimizer = AdamW(
            full_model.parameters(),
            lr=float(config["full_finetune_learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        full_training = train_with_validation(
            full_model,
            target_train,
            target_train_token_ids,
            target_validation,
            target_concept_token_ids,
            full_optimizer,
            int(config["target_epochs"]),
            config,
            device,
            trial_seed,
        )
        full_target = evaluate(
            full_model,
            target_test,
            target_concept_token_ids,
            config,
            device,
        )
        full_source = evaluate(
            full_model,
            source_test,
            source_concept_token_ids,
            config,
            device,
        )
        trial_result["full_finetune"] = {
            "target": metric_dictionary(full_target),
            "source": metric_dictionary(full_source),
            "best_epoch": full_training["best_epoch"],
            "best_validation_loss": full_training["best_validation_loss"],
        }
        trials.append(trial_result)
        if trial_index == 0:
            first_trial_training = {
                "lora": lora_training,
                "full_finetune": full_training,
            }
            save_lora_checkpoint(
                OUTPUT_DIR / "trial_1_lora_model.pt", lora_model, config,
                vocabulary, source_concepts, lora_training,
            )

        logger.info(
            "试验 %d：Frozen %.2f%%，LoRA %.2f%%，Full %.2f%%",
            trial_index + 1,
            base_target.image_to_text_accuracy * 100.0,
            lora_target.image_to_text_accuracy * 100.0,
            full_target.image_to_text_accuracy * 100.0,
        )

    target_aggregate = aggregate_trials(
        trials,
        "target",
        "image_to_text_accuracy",
    )
    source_aggregate = aggregate_trials(
        trials,
        "source",
        "image_to_text_accuracy",
    )
    save_accuracy_comparison(target_aggregate, source_aggregate)
    save_parameter_comparison(trainable_counts)
    save_first_trial_curves(first_trial_training)
    results = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "device": str(device),
        "source_descriptions": list(source_concepts),
        "target_descriptions": list(TARGET_DESCRIPTIONS),
        "sample_counts": {
            "source_train": int(source_train.images.shape[0]),
            "source_validation": int(source_validation.images.shape[0]),
            "source_test": int(source_test.images.shape[0]),
            "target_train_per_trial": int(
                int(config["target_train_samples_per_concept"])
                * len(TARGET_DESCRIPTIONS)
            ),
            "target_validation": int(target_validation.images.shape[0]),
            "target_test": int(target_test.images.shape[0]),
        },
        "base_parameter_count": base_parameter_count,
        "trainable_parameter_counts": trainable_counts,
        "lora_trainable_fraction": (
            trainable_counts["lora"]
            / (base_parameter_count + trainable_counts["lora"])
        ),
        "source_pretraining": {
            "best_epoch": source_training["best_epoch"],
            "best_validation_loss": source_training["best_validation_loss"],
            "source_test": metric_dictionary(base_source),
            "target_alias_before_adaptation": metric_dictionary(base_target),
        },
        "trials": trials,
        "target_accuracy_aggregate": target_aggregate,
        "source_retention_accuracy_aggregate": source_aggregate,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    logger.info("设备：%s", device)
    logger.info("基础模型参数量：%d", base_parameter_count)
    logger.info(
        "可训练参数：Frozen %d，LoRA %d，Full %d",
        trainable_counts["frozen"],
        trainable_counts["lora"],
        trainable_counts["full_finetune"],
    )
    for method_name in METHOD_NAMES:
        logger.info(
            "%s：目标 %.2f%% ± %.2f%%，来源保持 %.2f%% ± %.2f%%",
            method_name,
            target_aggregate[method_name]["mean"] * 100.0,
            target_aggregate[method_name]["standard_deviation"] * 100.0,
            source_aggregate[method_name]["mean"] * 100.0,
            source_aggregate[method_name]["standard_deviation"] * 100.0,
        )
    logger.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_inference_cli("lora_dual_encoder")
    else:
        main()
