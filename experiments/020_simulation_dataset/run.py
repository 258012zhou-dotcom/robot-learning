"""Collect, split, save, reload, and inspect a small simulation dataset."""

from dataclasses import fields
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from robot_learning.gymnasium_rollout import (
    ProportionalReachPolicy,
    RandomPolicy,
    run_episode,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.trajectory_dataset import (
    CollectedEpisode,
    DATASET_SCHEMA_VERSION,
    TEST_SPLIT_ID,
    TRAIN_SPLIT_ID,
    TransitionDataset,
    VALIDATION_SPLIT_ID,
    assign_stratified_episode_splits,
    build_transition_dataset,
    load_transition_dataset,
    save_transition_dataset,
    validate_transition_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "020_simulation_dataset.json"
MODEL_PATH = (
    PROJECT_ROOT / "experiments" / "017_mujoco_step" / "point_robot.xml"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "020_simulation_dataset"
DATASET_PATH = OUTPUT_DIR / "dataset.npz"
POLICY_NAMES = {0: "random", 1: "proportional"}
SPLIT_NAMES = {
    TRAIN_SPLIT_ID: "train",
    VALIDATION_SPLIT_ID: "validation",
    TEST_SPLIT_ID: "test",
}


def load_config() -> dict[str, Any]:
    """Read settings and validate counts needed by the three-way split."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    if int(config["episode_count_per_policy"]) < 3:
        raise ValueError("each policy needs at least three Episodes")
    return config


def make_environment(config: dict[str, Any]) -> PointRobotReachEnv:
    """Construct the same environment used by both collection policies."""
    return PointRobotReachEnv(
        MODEL_PATH,
        frame_skip=int(config["frame_skip"]),
        max_episode_steps=int(config["max_episode_steps"]),
        minimum_target_distance=float(config["minimum_target_distance"]),
        maximum_target_distance=float(config["maximum_target_distance"]),
        success_tolerance=float(config["success_tolerance"]),
        velocity_tolerance=float(config["velocity_tolerance"]),
        action_penalty_weight=float(config["action_penalty_weight"]),
    )


def collect_episodes(
    config: dict[str, Any],
    environment: PointRobotReachEnv,
) -> list[CollectedEpisode]:
    """Collect equal Episode counts from random and successful baselines."""
    first_seed = int(config["seed"])
    count = int(config["episode_count_per_policy"])
    episode_ids = list(range(2 * count))
    policy_ids = [0] * count + [1] * count
    split_assignments = assign_stratified_episode_splits(
        episode_ids,
        policy_ids,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
        seed=first_seed,
    )

    policies = {
        0: RandomPolicy(
            environment.action_space,
            seed=first_seed + int(config["random_policy_seed_offset"]),
        ),
        1: ProportionalReachPolicy(
            float(config["proportional_gain"]),
            environment.action_space,
        ),
    }
    episodes: list[CollectedEpisode] = []
    for policy_id, policy in policies.items():
        for local_index in range(count):
            episode_id = policy_id * count + local_index
            environment_seed = first_seed + local_index
            result = run_episode(
                environment,
                policy,
                seed=environment_seed,
            )
            episodes.append(
                CollectedEpisode(
                    episode_id=episode_id,
                    environment_seed=environment_seed,
                    policy_id=policy_id,
                    split_id=split_assignments[episode_id],
                    result=result,
                )
            )

    # Matching environment seeds must generate matching targets across policies.
    random_targets = [
        episode.result.observations[0, 2] for episode in episodes[:count]
    ]
    proportional_targets = [
        episode.result.observations[0, 2] for episode in episodes[count:]
    ]
    if not np.array_equal(random_targets, proportional_targets):
        raise RuntimeError("collection policies received different targets")
    return episodes


def assert_round_trip_equal(
    original: TransitionDataset,
    loaded: TransitionDataset,
) -> None:
    """Require every saved and reloaded array to match exactly."""
    for field in fields(original):
        if not np.array_equal(
            getattr(original, field.name),
            getattr(loaded, field.name),
        ):
            raise RuntimeError(f"NPZ round trip changed {field.name}")


def dataset_content_hash(dataset: TransitionDataset) -> str:
    """Hash array names, dtypes, shapes, and bytes independent of ZIP metadata."""
    digest = hashlib.sha256()
    for field in fields(dataset):
        array = np.ascontiguousarray(getattr(dataset, field.name))
        digest.update(field.name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def episode_statistics(
    episodes: list[CollectedEpisode],
    *,
    policy_id: int | None = None,
    split_id: int | None = None,
) -> dict[str, int | float]:
    """Summarize selected complete Episodes rather than individual rows."""
    selected = [
        episode
        for episode in episodes
        if (policy_id is None or episode.policy_id == policy_id)
        and (split_id is None or episode.split_id == split_id)
    ]
    successes = sum(episode.result.is_success for episode in selected)
    return {
        "episode_count": len(selected),
        "transition_count": sum(
            episode.result.step_count for episode in selected
        ),
        "success_count": successes,
        "success_rate": successes / len(selected) if selected else 0.0,
    }


def build_manifest(
    config: dict[str, Any],
    environment: PointRobotReachEnv,
    episodes: list[CollectedEpisode],
    dataset: TransitionDataset,
) -> dict[str, Any]:
    """Describe schema, provenance, split membership, and collection quality."""
    return {
        "experiment_name": config["experiment_name"],
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_content_sha256": dataset_content_hash(dataset),
        "source_model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "transition_count": dataset.transition_count,
        "episode_count": dataset.episode_count,
        "array_schema": {
            field.name: {
                "shape": list(getattr(dataset, field.name).shape),
                "dtype": str(getattr(dataset, field.name).dtype),
            }
            for field in fields(dataset)
        },
        "observation_fields": [
            "position",
            "velocity",
            "target_position",
            "target_error",
        ],
        "action_fields": ["motor_control"],
        "action_range": [
            float(environment.action_space.low[0]),
            float(environment.action_space.high[0]),
        ],
        "control_timestep": environment.control_timestep,
        "policy_ids": {str(key): value for key, value in POLICY_NAMES.items()},
        "split_ids": {str(key): value for key, value in SPLIT_NAMES.items()},
        "policy_statistics": {
            name: episode_statistics(episodes, policy_id=policy_id)
            for policy_id, name in POLICY_NAMES.items()
        },
        "split_statistics": {
            name: episode_statistics(episodes, split_id=split_id)
            for split_id, name in SPLIT_NAMES.items()
        },
        "split_episode_ids": {
            name: sorted(
                episode.episode_id
                for episode in episodes
                if episode.split_id == split_id
            )
            for split_id, name in SPLIT_NAMES.items()
        },
    }


def save_distribution_plot(dataset: TransitionDataset) -> None:
    """Show how collection policy changes state and action distributions."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for policy_id, policy_name in POLICY_NAMES.items():
        rows = dataset.policy_ids == policy_id
        axes[0].hist(
            dataset.observations[rows, 3],
            bins=35,
            density=True,
            alpha=0.55,
            label=policy_name,
        )
        axes[1].hist(
            dataset.actions[rows, 0],
            bins=35,
            density=True,
            alpha=0.55,
            label=policy_name,
        )
    axes[0].set(xlabel="target error [m]", ylabel="density")
    axes[1].set(xlabel="motor action", ylabel="density")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Transition distributions depend on collection policy")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "dataset_distributions.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Collect reproducible trajectories and verify the stored dataset."""
    config = load_config()
    environment = make_environment(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=(
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "run.log", encoding="utf-8"),
        ),
    )

    episodes = collect_episodes(config, environment)
    dataset = build_transition_dataset(episodes)
    validate_transition_dataset(
        dataset,
        action_low=environment.action_space.low,
        action_high=environment.action_space.high,
    )
    save_transition_dataset(DATASET_PATH, dataset)

    # Validation after disk I/O catches serialization and schema mistakes.
    loaded_dataset = load_transition_dataset(DATASET_PATH)
    validate_transition_dataset(
        loaded_dataset,
        action_low=environment.action_space.low,
        action_high=environment.action_space.high,
    )
    assert_round_trip_equal(dataset, loaded_dataset)

    manifest = build_manifest(config, environment, episodes, loaded_dataset)
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    save_distribution_plot(loaded_dataset)

    logging.info(
        "数据集：%s Episodes，%s Transitions",
        manifest["episode_count"],
        manifest["transition_count"],
    )
    for policy_name, statistics in manifest["policy_statistics"].items():
        logging.info(
            "%s：%s Episodes，%s Transitions，成功率 %.1f%%",
            policy_name,
            statistics["episode_count"],
            statistics["transition_count"],
            100.0 * statistics["success_rate"],
        )
    logging.info("NPZ 保存后无损读取，内容哈希：%s", manifest["dataset_content_sha256"])
    logging.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
