"""Run a reproducible visual preprocessing pipeline."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from robot_learning.vision_preprocessing import (
    CameraIntrinsics,
    bgr_image_to_batch_tensor,
    detect_largest_hsv_target,
    letterbox_image,
    map_bounding_box,
    transform_camera_intrinsics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "006_vision_preprocessing.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "006_vision_preprocessing"


def load_config() -> dict[str, Any]:
    """Load the experiment JSON configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def create_synthetic_scene(config: dict[str, Any]) -> np.ndarray:
    """Create a deterministic BGR camera frame with known target geometry."""
    image_width, image_height = map(int, config["image_size"])
    image = np.full(
        (image_height, image_width, 3),
        config["background_bgr"],
        dtype=np.uint8,
    )

    x, y, width, height = map(int, config["target_bbox"])
    image[y:y + height, x:x + width] = config["target_bgr"]

    # A smaller green region checks that the largest target is selected.
    image[60:85, 80:105] = config["target_bgr"]
    cv2.circle(image, (180, 350), 45, (0, 0, 255), thickness=-1)
    return image


def draw_bounding_box(
    image_bgr: np.ndarray,
    bounding_box: tuple[float, float, float, float],
) -> np.ndarray:
    """Return a copy with one yellow bounding box."""
    annotated = image_bgr.copy()
    x, y, width, height = (
        int(round(value)) for value in bounding_box
    )
    cv2.rectangle(
        annotated,
        (x, y),
        (x + width - 1, y + height - 1),
        (0, 255, 255),
        thickness=3,
    )
    return annotated


def save_image(path: Path, image: np.ndarray) -> None:
    """Save an image and fail clearly when OpenCV cannot write it."""
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to save image: {path}")


def main() -> None:
    """Detect, transform, convert, and save the visual pipeline outputs."""
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    scene = create_synthetic_scene(config)
    mask, detected_box = detect_largest_hsv_target(
        scene,
        tuple(config["lower_hsv"]),
        tuple(config["upper_hsv"]),
        minimum_area=float(config["minimum_area"]),
    )
    if detected_box is None:
        raise RuntimeError("the configured target was not detected")

    model_size = tuple(config["model_input_size"])
    letterboxed, transform = letterbox_image(scene, model_size)
    model_box = map_bounding_box(detected_box, transform)

    raw_intrinsics = CameraIntrinsics(**config["camera_intrinsics"])
    model_intrinsics = transform_camera_intrinsics(
        raw_intrinsics,
        transform,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_tensor = bgr_image_to_batch_tensor(
        letterboxed,
        device=device,
    )

    save_image(OUTPUT_DIR / "scene.png", scene)
    save_image(OUTPUT_DIR / "mask.png", mask)
    save_image(
        OUTPUT_DIR / "annotated_original.png",
        draw_bounding_box(scene, detected_box),
    )
    save_image(OUTPUT_DIR / "letterboxed.png", letterboxed)
    save_image(
        OUTPUT_DIR / "annotated_model_input.png",
        draw_bounding_box(letterboxed, model_box),
    )

    results = {
        "experiment_name": config["experiment_name"],
        "original_image_shape_hwc": list(scene.shape),
        "detected_bbox_xywh": list(detected_box),
        "model_bbox_xywh": list(model_box),
        "letterbox_transform": asdict(transform),
        "original_camera_intrinsics": asdict(raw_intrinsics),
        "model_camera_intrinsics": asdict(model_intrinsics),
        "mask_nonzero_pixels": int(np.count_nonzero(mask)),
        "model_tensor": {
            "shape_nchw": list(model_tensor.shape),
            "dtype": str(model_tensor.dtype),
            "device": str(model_tensor.device),
            "minimum": float(model_tensor.min().item()),
            "maximum": float(model_tensor.max().item()),
        },
    }
    with (OUTPUT_DIR / "results.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    logger.info("检测框（原图）：%s", detected_box)
    logger.info("检测框（模型输入）：%s", model_box)
    logger.info("模型张量：%s", results["model_tensor"])
    logger.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
