"""Self-contained inference checkpoints for experiments 008, 009, and 015."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import cv2
import torch
from torch import nn

from robot_learning.object_detection import SmallShapeDetector
from robot_learning.peft import inject_lora_into_linear_layers
from robot_learning.semantic_segmentation import SmallSemanticSegmenter
from robot_learning.vision_language import SimpleVocabulary, VisionLanguageDualEncoder
from robot_learning.vision_preprocessing import bgr_image_to_batch_tensor


def image_preprocessing(image_size: int) -> dict[str, Any]:
    """Describe the existing generators' input contract; no resizing is hidden."""
    return {
        "image_size": image_size,
        "color_order": "RGB",
        "layout": "NCHW",
        "dtype": "float32",
        "scale": 1.0 / 255.0,
        "normalization": "uint8 / 255; no mean/std normalization",
        "resize": "none; require exact square image_size",
    }


def save_model_artifact(
    path: Path, model: nn.Module, metadata: dict[str, Any]
) -> None:
    """Save all weights (including frozen base weights) and plain metadata."""
    artifact = deepcopy(metadata)
    artifact["format_version"] = 1
    # Clone avoids sharing CPU storage with the live model.
    artifact["state_dict"] = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    torch.save(artifact, path)


def load_model_artifact(
    path: Path, device: str | torch.device = "cpu", *, expected_kind: str | None = None
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild known architectures and strictly load every parameter/buffer."""
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format_version (legacy adapter is incomplete)")
    kind = artifact["model_kind"]
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"expected {expected_kind} checkpoint, got {kind}")
    architecture = artifact["architecture"]
    preprocessing = artifact["preprocessing"]
    size = preprocessing["image_size"]
    if type(size) is not int or size < 4 or preprocessing != image_preprocessing(size):
        raise ValueError("unsupported image preprocessing contract")
    if kind in ("shape_detector", "semantic_segmenter"):
        classes = artifact["class_names"]
        if len(classes) != architecture["num_classes"] or len(set(classes)) != len(classes):
            raise ValueError("class order must contain one unique name per output class")
        model_type = SmallShapeDetector if kind == "shape_detector" else SmallSemanticSegmenter
        model = model_type(**architecture)
    elif kind == "lora_dual_encoder":
        tokens = artifact["vocabulary_tokens"]
        if (
            len(tokens) != architecture["vocabulary_size"]
            or len(set(tokens)) != len(tokens)
            or tokens[:2] != ["<PAD>", "<UNK>"]
            or architecture["padding_id"] != 0
        ):
            raise ValueError("invalid vocabulary order or padding ID")
        if artifact["tokenization"] != "lowercase [a-z0-9]+; right pad; reject overlength":
            raise ValueError("unsupported tokenization")
        model = VisionLanguageDualEncoder(**architecture)
        # Injection changes state_dict names; it must precede strict loading.
        inject_lora_into_linear_layers(model, **artifact["lora"])
    else:
        raise ValueError(f"unsupported model_kind: {kind}")
    model.load_state_dict(artifact["state_dict"], strict=True)
    # Move the complete reconstructed model, including newly created adapters.
    model.to(device)
    model.eval()
    return model, artifact


def run_inference_cli(expected_kind: str) -> None:
    """Load a checkpoint and image without accessing training configuration."""
    parser = argparse.ArgumentParser(description="Train by default, or load for inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, artifact = load_model_artifact(
        args.checkpoint, args.device, expected_kind=expected_kind
    )
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {args.image}")
    size = artifact["preprocessing"]["image_size"]
    if image.shape[:2] != (size, size):
        raise ValueError(f"image must be {size} x {size}; resize explicitly before inference")
    images = bgr_image_to_batch_tensor(image, device=args.device)
    with torch.inference_mode():
        if expected_kind == "shape_detector":
            logits, boxes = model(images)
            index = int(logits.argmax(dim=1).item())
            result = {
                "class_index": index, "class_name": artifact["class_names"][index],
                "box_format": "normalized_cxcywh", "box": boxes[0].cpu().tolist(),
            }
        elif expected_kind == "semantic_segmenter":
            result = {
                "class_names": artifact["class_names"],
                "mask": model(images).argmax(dim=1)[0].cpu().tolist(),
            }
        else:
            vocabulary = SimpleVocabulary({
                token: index for index, token in enumerate(artifact["vocabulary_tokens"])
            })
            descriptions = artifact["target_descriptions"]
            tokens = vocabulary.encode_batch(
                descriptions, artifact["architecture"]["maximum_token_count"]
            ).to(args.device)
            embeddings = model(images, tokens)
            similarities = embeddings.image_embeddings @ embeddings.text_embeddings.T
            index = int(similarities.argmax(dim=1).item())
            result = {
                "descriptions": descriptions, "prediction": descriptions[index],
                "cosine_similarities": similarities[0].cpu().tolist(),
            }
    print(json.dumps(result, ensure_ascii=False))
