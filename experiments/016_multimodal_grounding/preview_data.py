"""Preview language-conditioned grounding samples before model training.

Reading order:
1. ``load_config`` reads the small JSON experiment contract.
2. ``plot_sample`` explains one scene on one Matplotlib axis.
3. ``main`` connects data generation, plotting, and output saving.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from robot_learning.multimodal_grounding import (
    GroundingDataset,
    generate_grounding_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "016_multimodal_grounding.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "016_multimodal_grounding"
    / "data_preview.png"
)


def load_config() -> dict[str, Any]:
    """Read preview settings from the experiment JSON file."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def plot_sample(
    axis: plt.Axes,
    dataset: GroundingDataset,
    sample_index: int,
) -> None:
    """Draw one scene and distinguish its target from distractors."""
    # PyTorch stores images as (channels, height, width), whereas Matplotlib
    # expects (height, width, channels).
    image = dataset.images[sample_index].permute(1, 2, 0).numpy()
    image_height, image_width = image.shape[:2]
    target_index = int(dataset.target_indices[sample_index])

    axis.imshow(image)
    for object_index, normalized_center in enumerate(
        dataset.scene_centers[sample_index]
    ):
        center_x = float(normalized_center[0]) * image_width
        center_y = float(normalized_center[1]) * image_height

        if object_index == target_index:
            # A hollow red circle marks the object requested by the instruction.
            axis.scatter(
                center_x,
                center_y,
                s=180,
                facecolors="none",
                edgecolors="red",
                linewidths=2.0,
                label="target",
            )
        else:
            # Gray crosses expose all distractor centers without hiding objects.
            axis.scatter(
                center_x,
                center_y,
                s=45,
                marker="x",
                color="lightgray",
                linewidths=1.5,
            )

    axis.set_title(dataset.instructions[sample_index], fontsize=10)
    axis.set_axis_off()


def main() -> None:
    """Generate a small deterministic batch and save a labeled preview grid."""
    config = load_config()
    sample_count = int(config["preview_sample_count"])
    dataset = generate_grounding_dataset(
        sample_count=sample_count,
        image_size=int(config["image_size"]),
        object_count=int(config["object_count"]),
        seed=int(config["seed"]),
    )

    # Two rows keep the image and instruction text readable for six samples.
    column_count = 3
    row_count = (sample_count + column_count - 1) // column_count
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(10, 3.4 * row_count),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for sample_index in range(sample_count):
        plot_sample(flat_axes[sample_index], dataset, sample_index)
    for unused_axis in flat_axes[sample_count:]:
        unused_axis.set_visible(False)

    figure.suptitle(
        "Language-conditioned grounding data\n"
        "red circle = requested target, gray cross = distractor",
        fontsize=13,
    )
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Generated {sample_count} scenes")
    print(f"Preview saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
