"""Visualize synthetic shape samples before training the CNN."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from robot_learning.image_classification import (
    generate_shape_classification_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "007_cnn_classification.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "007_cnn_classification"
CLASS_NAMES = ("square", "circle")


def load_config() -> dict[str, Any]:
    """Load the experiment JSON configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """Generate deterministic samples and save a labeled image grid."""
    config = load_config()
    images, labels = generate_shape_classification_dataset(
        sample_count=12,
        image_size=int(config["image_size"]),
        seed=int(config["seed"]),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 4, figsize=(8, 6))
    for axis, image, label in zip(axes.flat, images, labels):
        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_title(f"label={int(label)}: {CLASS_NAMES[int(label)]}")
        axis.axis("off")
    figure.suptitle("Synthetic Shape Classification Samples")
    figure.tight_layout()
    output_path = OUTPUT_DIR / "dataset_preview.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    print(f"样本张量形状：{tuple(images.shape)}")
    print(f"标签数量：square={int((labels == 0).sum())}, "
          f"circle={int((labels == 1).sum())}")
    print(f"预览图：{output_path}")


if __name__ == "__main__":
    main()
