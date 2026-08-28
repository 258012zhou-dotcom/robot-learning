"""Create a synthetic RGB-D scene and backproject semantic point clouds."""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from robot_learning.depth_point_cloud import (
    PointCloud,
    camera_points_to_pixels,
    depth_image_to_point_cloud,
)
from robot_learning.vision_preprocessing import CameraIntrinsics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "010_depth_point_cloud.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "010_depth_point_cloud"
CLASS_NAMES = ("background", "square", "circle")
MASK_COLORS = np.array(
    [[0, 0, 0], [50, 220, 80], [255, 150, 40]],
    dtype=np.uint8,
)


def load_config() -> dict[str, Any]:
    """Load the experiment JSON configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def generate_rgbd_scene(
    config: dict[str, Any],
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Generate RGB, Z-depth in meters, and semantic class mask."""
    seed = int(config["seed"])
    image_width = int(config["image_width"])
    image_height = int(config["image_height"])
    rng = np.random.default_rng(seed)

    image_bgr = rng.integers(
        0,
        31,
        size=(image_height, image_width, 3),
        dtype=np.uint8,
    )
    semantic_mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8,
    )
    ideal_depth = np.full(
        (image_height, image_width),
        float(config["background_depth_m"]),
        dtype=np.float32,
    )

    square_left = image_width // 8
    square_right = image_width * 7 // 16
    square_top = image_height // 5
    square_bottom = image_height * 3 // 5
    cv2.rectangle(
        image_bgr,
        (square_left, square_top),
        (square_right - 1, square_bottom - 1),
        (80, 230, 100),
        thickness=-1,
    )
    cv2.rectangle(
        semantic_mask,
        (square_left, square_top),
        (square_right - 1, square_bottom - 1),
        1,
        thickness=-1,
    )
    ideal_depth[semantic_mask == 1] = float(config["square_depth_m"])

    circle_center = (image_width * 3 // 4, image_height * 3 // 5)
    circle_radius = image_height // 5
    cv2.circle(
        image_bgr,
        circle_center,
        circle_radius,
        (40, 150, 255),
        thickness=-1,
    )
    cv2.circle(
        semantic_mask,
        circle_center,
        circle_radius,
        2,
        thickness=-1,
    )
    ideal_depth[semantic_mask == 2] = float(config["circle_depth_m"])

    noise = rng.normal(
        0.0,
        float(config["depth_noise_standard_deviation_m"]),
        size=ideal_depth.shape,
    ).astype(np.float32)
    depth_m = ideal_depth + noise
    invalid_mask = rng.random(ideal_depth.shape) < float(
        config["invalid_depth_fraction"]
    )
    depth_m[invalid_mask] = 0.0
    depth_m[0, 0] = np.nan
    depth_m[0, 1] = np.inf

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return (
        image_rgb,
        torch.from_numpy(depth_m),
        torch.from_numpy(semantic_mask).to(dtype=torch.int64),
    )


def point_cloud_statistics(cloud: PointCloud) -> dict[str, Any]:
    """Return JSON-compatible count, centroid, and axis ranges."""
    if cloud.points_m.shape[0] == 0:
        return {
            "point_count": 0,
            "centroid_m": None,
            "minimum_xyz_m": None,
            "maximum_xyz_m": None,
        }
    return {
        "point_count": cloud.points_m.shape[0],
        "centroid_m": cloud.points_m.mean(dim=0).tolist(),
        "minimum_xyz_m": cloud.points_m.min(dim=0).values.tolist(),
        "maximum_xyz_m": cloud.points_m.max(dim=0).values.tolist(),
    }


def save_scene_overview(
    image_rgb: np.ndarray,
    depth_m: torch.Tensor,
    semantic_mask: torch.Tensor,
    minimum_depth_m: float,
    maximum_depth_m: float,
) -> None:
    """Save RGB, depth, semantic mask, and valid-depth views."""
    depth_display = depth_m.numpy().copy()
    depth_display[~np.isfinite(depth_display)] = np.nan
    valid_depth = (
        torch.isfinite(depth_m)
        & (depth_m >= minimum_depth_m)
        & (depth_m <= maximum_depth_m)
        & (depth_m > 0.0)
    )
    figure, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(image_rgb)
    axes[0].set_title("RGB image")
    depth_image = axes[1].imshow(
        depth_display,
        cmap="viridis",
        vmin=minimum_depth_m,
        vmax=maximum_depth_m,
    )
    axes[1].set_title("Z-depth (meters)")
    figure.colorbar(depth_image, ax=axes[1], fraction=0.046)
    axes[2].imshow(MASK_COLORS[semantic_mask.numpy()])
    axes[2].set_title("semantic mask")
    axes[3].imshow(valid_depth.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("valid depth mask")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "scene_overview.png", dpi=150)
    plt.close(figure)


def save_point_cloud_plot(
    full_cloud: PointCloud,
    square_cloud: PointCloud,
    circle_cloud: PointCloud,
    seed: int,
) -> None:
    """Save a camera-frame three-dimensional point-cloud visualization."""
    rng = np.random.default_rng(seed)
    background_count = min(3000, full_cloud.points_m.shape[0])
    background_indices = rng.choice(
        full_cloud.points_m.shape[0],
        size=background_count,
        replace=False,
    )
    background = full_cloud.points_m[background_indices].numpy()
    square = square_cloud.points_m.numpy()
    circle = circle_cloud.points_m.numpy()

    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        background[:, 0],
        background[:, 1],
        background[:, 2],
        s=1,
        alpha=0.08,
        color="gray",
        label="all valid depth (sampled)",
    )
    axis.scatter(
        square[:, 0],
        square[:, 1],
        square[:, 2],
        s=3,
        alpha=0.6,
        color="limegreen",
        label="square mask points",
    )
    axis.scatter(
        circle[:, 0],
        circle[:, 1],
        circle[:, 2],
        s=3,
        alpha=0.6,
        color="darkorange",
        label="circle mask points",
    )
    axis.set(
        title="Camera optical-frame point clouds",
        xlabel="X right (m)",
        ylabel="Y down (m)",
        zlabel="Z forward (m)",
    )
    axis.legend()
    axis.view_init(elev=24, azim=-65)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "point_clouds.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Generate the scene, backproject point clouds, and save evidence."""
    config = load_config()
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
    image_rgb, depth_m, semantic_mask = generate_rgbd_scene(config)
    intrinsics_config = config["camera_intrinsics"]
    intrinsics = CameraIntrinsics(
        fx=float(intrinsics_config["fx"]),
        fy=float(intrinsics_config["fy"]),
        cx=float(intrinsics_config["cx"]),
        cy=float(intrinsics_config["cy"]),
    )
    minimum_depth_m = float(config["minimum_depth_m"])
    maximum_depth_m = float(config["maximum_depth_m"])
    cloud_arguments = {
        "intrinsics": intrinsics,
        "minimum_depth_m": minimum_depth_m,
        "maximum_depth_m": maximum_depth_m,
    }
    full_cloud = depth_image_to_point_cloud(
        depth_m,
        **cloud_arguments,
    )
    square_cloud = depth_image_to_point_cloud(
        depth_m,
        selection_mask=semantic_mask == 1,
        **cloud_arguments,
    )
    circle_cloud = depth_image_to_point_cloud(
        depth_m,
        selection_mask=semantic_mask == 2,
        **cloud_arguments,
    )

    recovered_pixels = camera_points_to_pixels(
        full_cloud.points_m,
        intrinsics,
    )
    projection_errors = torch.linalg.vector_norm(
        recovered_pixels - full_cloud.pixels_uv.to(dtype=torch.float32),
        dim=1,
    )
    maximum_projection_error_pixels = float(projection_errors.max().item())
    full_statistics = point_cloud_statistics(full_cloud)
    square_statistics = point_cloud_statistics(square_cloud)
    circle_statistics = point_cloud_statistics(circle_cloud)
    if square_statistics["point_count"] == 0 or circle_statistics["point_count"] == 0:
        raise RuntimeError("target semantic point clouds must not be empty")
    square_depth = square_statistics["centroid_m"][2]
    circle_depth = circle_statistics["centroid_m"][2]
    if not square_depth < circle_depth < float(config["background_depth_m"]):
        raise RuntimeError("target point-cloud depth ordering is incorrect")
    if maximum_projection_error_pixels > 1e-4:
        raise RuntimeError("projection round-trip error exceeds tolerance")

    save_scene_overview(
        image_rgb,
        depth_m,
        semantic_mask,
        minimum_depth_m,
        maximum_depth_m,
    )
    save_point_cloud_plot(
        full_cloud,
        square_cloud,
        circle_cloud,
        seed=int(config["seed"]),
    )
    np.savez_compressed(
        OUTPUT_DIR / "target_point_clouds.npz",
        square_points_m=square_cloud.points_m.numpy(),
        circle_points_m=circle_cloud.points_m.numpy(),
        square_pixels_uv=square_cloud.pixels_uv.numpy(),
        circle_pixels_uv=circle_cloud.pixels_uv.numpy(),
    )

    results = {
        "experiment_name": config["experiment_name"],
        "seed": int(config["seed"]),
        "image_size": [int(config["image_height"]), int(config["image_width"])],
        "camera_intrinsics": {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
        },
        "depth_unit": "meters",
        "maximum_projection_round_trip_error_pixels": (
            maximum_projection_error_pixels
        ),
        "full_valid_cloud": full_statistics,
        "square_cloud": square_statistics,
        "circle_cloud": circle_statistics,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    logger.info("完整有效点数：%s", full_statistics["point_count"])
    logger.info(
        "正方形点云：%s 点，中心 %s",
        square_statistics["point_count"],
        np.round(square_statistics["centroid_m"], 4),
    )
    logger.info(
        "圆形点云：%s 点，中心 %s",
        circle_statistics["point_count"],
        np.round(circle_statistics["centroid_m"], 4),
    )
    logger.info(
        "最大投影往返误差：%.8f pixels",
        maximum_projection_error_pixels,
    )
    logger.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
