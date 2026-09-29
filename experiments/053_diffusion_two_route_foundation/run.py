"""Validate both expert routes and store a scene-separated teaching dataset."""

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from robot_learning.gymnasium_rollout import EpisodeResult, run_episode
from robot_learning.trajectory_dataset import (
    CollectedEpisode,
    TEST_SPLIT_ID,
    TRAIN_SPLIT_ID,
    TransitionDataset,
    VALIDATION_SPLIT_ID,
    build_transition_dataset,
    load_transition_dataset,
    save_transition_dataset,
    validate_transition_dataset,
)
from robot_learning.two_route_navigation import (
    TwoRouteExpert,
    TwoRouteNavigationEnv,
    segment_clearance,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/053_diffusion_two_route_foundation.json"
OUTPUT_DIR = ROOT / "outputs/053_diffusion_two_route_foundation"
SPLIT_IDS = {"train": TRAIN_SPLIT_ID, "validation": VALIDATION_SPLIT_ID, "test": TEST_SPLIT_ID}
ROUTES = {0: -1, 1: 1}  # lower, upper


def scene_seed_splits(config: dict[str, Any]) -> dict[str, list[int]]:
    """Assign both routes of one scene to the same immutable split."""
    ranges = {}
    for name in SPLIT_IDS:
        settings = config["scene_splits"][name]
        start, count = int(settings["seed_start"]), int(settings["scene_count"])
        if start < 0 or count <= 0:
            raise ValueError("scene seed starts and counts must be valid")
        ranges[name] = list(range(start, start + count))
    all_seeds = [seed for seeds in ranges.values() for seed in seeds]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("scene seed splits overlap")
    return ranges


def episode_clearance(observations: np.ndarray) -> float:
    """Minimum actual segment clearance, including between logged endpoints."""
    center = observations[0, 4:6]
    radius = float(observations[0, 6])
    return min(
        segment_clearance(first[:2], second[:2], center, radius)
        for first, second in zip(observations[:-1], observations[1:], strict=True)
    )


def audit_scene_groups(dataset: TransitionDataset, seed_splits: dict[str, list[int]]) -> None:
    """Catch scene leakage that ordinary per-Episode validation cannot detect."""
    for split_name, seeds in seed_splits.items():
        for seed in seeds:
            rows = np.flatnonzero(dataset.environment_seeds == seed)
            if rows.size == 0:
                raise RuntimeError("a configured scene has no saved transitions")
            if set(dataset.policy_ids[rows].tolist()) != set(ROUTES):
                raise RuntimeError("a scene is missing one of its two routes")
            if set(dataset.split_ids[rows].tolist()) != {SPLIT_IDS[split_name]}:
                raise RuntimeError("the routes of one scene cross data splits")
            if np.unique(dataset.episode_ids[rows]).size != len(ROUTES):
                raise RuntimeError("a scene must contain exactly two Episodes")


def save_route_preview(scene_results: dict[int, EpisodeResult]) -> None:
    """Show the same initial scene with two successful expert paths."""
    initial = scene_results[-1].observations[0]
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.add_patch(
        Circle(initial[4:6], float(initial[6]), color="gray", alpha=0.4, label="obstacle")
    )
    route_styles = (
        (-1, "tab:blue", "lower expert"),
        (1, "tab:orange", "upper expert"),
    )
    for route, color, label in route_styles:
        positions = scene_results[route].observations[:, :2]
        axes.plot(positions[:, 0], positions[:, 1], color=color, label=label)
    axes.scatter([initial[0], initial[2]], [initial[1], initial[3]], c=["green", "red"], zorder=3)
    axes.annotate("start", initial[:2])
    axes.annotate("goal", initial[2:4])
    axes.set(xlabel="x (m)", ylabel="y (m)", title="Two expert routes in the same scene")
    axes.set_aspect("equal")
    axes.grid(alpha=0.2)
    axes.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "route_preview.png", dpi=150)
    plt.close(figure)


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    seed_splits = scene_seed_splits(config)
    environment = TwoRouteNavigationEnv(
        dt=float(config["dt"]),
        max_episode_steps=int(config["max_episode_steps"]),
        goal_tolerance=float(config["goal_tolerance"]),
    )
    collected: list[CollectedEpisode] = []
    records: list[dict[str, Any]] = []
    direct_clearances: list[float] = []
    preview_results: dict[int, EpisodeResult] | None = None
    episode_id = 0
    try:
        for split_name, seeds in seed_splits.items():
            for seed in seeds:
                scene_results = {}
                for policy_id, route in ROUTES.items():
                    expert = TwoRouteExpert(
                        route, dt=environment.dt, speed=float(config["expert_speed"])
                    )
                    result = run_episode(environment, expert, seed=seed)
                    if not result.is_success or not result.terminated or result.truncated:
                        raise RuntimeError(
                            f"expert failed: split={split_name} seed={seed} route={route}"
                        )
                    clearance = episode_clearance(result.observations)
                    if clearance <= 0.0:
                        raise RuntimeError("a recorded successful route intersects the obstacle")
                    obstacle_y = float(result.observations[0, 5])
                    radius = float(result.observations[0, 6])
                    if route < 0 and not np.min(result.observations[:, 1]) < obstacle_y - radius:
                        raise RuntimeError("lower-route expert did not go below the obstacle")
                    if route > 0 and not np.max(result.observations[:, 1]) > obstacle_y + radius:
                        raise RuntimeError("upper-route expert did not go above the obstacle")
                    collected.append(
                        CollectedEpisode(
                            episode_id=episode_id,
                            environment_seed=seed,
                            policy_id=policy_id,
                            split_id=SPLIT_IDS[split_name],
                            result=result,
                        )
                    )
                    records.append(
                        {
                            "scene_seed": seed,
                            "split": split_name,
                            "route": "lower" if route < 0 else "upper",
                            "episode_id": episode_id,
                            "steps": result.step_count,
                            "success": result.is_success,
                            "minimum_obstacle_clearance": clearance,
                        }
                    )
                    scene_results[route] = result
                    episode_id += 1
                if not np.array_equal(
                    scene_results[-1].observations[0], scene_results[1].observations[0]
                ):
                    raise RuntimeError("the two routes did not start from the same scene")
                if preview_results is None:
                    preview_results = scene_results
                initial = scene_results[-1].observations[0]
                direct_clearance = segment_clearance(
                    initial[:2], initial[2:4], initial[4:6], float(initial[6])
                )
                if direct_clearance >= 0.0:
                    raise RuntimeError("the straight path is not blocked in this scene")
                direct_clearances.append(direct_clearance)
    finally:
        environment.close()

    dataset = build_transition_dataset(collected)
    validate_transition_dataset(
        dataset,
        action_low=np.asarray([-1.0, -1.0]),
        action_high=np.asarray([1.0, 1.0]),
    )
    audit_scene_groups(dataset, seed_splits)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if preview_results is None:
        raise RuntimeError("no scene was collected")
    save_route_preview(preview_results)
    dataset_path = OUTPUT_DIR / "dataset.npz"
    save_transition_dataset(dataset_path, dataset)
    loaded = load_transition_dataset(dataset_path)
    if (
        loaded.transition_count != dataset.transition_count
        or loaded.episode_count != dataset.episode_count
    ):
        raise RuntimeError("saved and reloaded dataset counts disagree")
    report = {
        "experiment_name": config["experiment_name"],
        "status": "expert_and_data_foundation_only_no_learned_policy",
        "source_type": "synthetic_2d_navigation_simulation",
        "observation": "[x, y, goal_x, goal_y, obstacle_x, obstacle_y, obstacle_radius] (m)",
        "action": "[vx, vy] (m/s)",
        "scene_splits": {
            name: {
                "scene_count": len(seeds),
                "seed_start": seeds[0],
                "seed_end": seeds[-1],
            }
            for name, seeds in seed_splits.items()
        },
        "route_count_per_scene": len(ROUTES),
        "episode_count": dataset.episode_count,
        "transition_count": dataset.transition_count,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "minimum_obstacle_clearance": min(row["minimum_obstacle_clearance"] for row in records),
        "maximum_straight_path_clearance": max(direct_clearances),
        "mean_episode_steps": float(np.mean([row["steps"] for row in records])),
        "episodes": records,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(
        f"{dataset.episode_count} expert Episodes, {dataset.transition_count} transitions; "
        f"minimum clearance={report['minimum_obstacle_clearance']:.4f} m"
    )


if __name__ == "__main__":
    main()
