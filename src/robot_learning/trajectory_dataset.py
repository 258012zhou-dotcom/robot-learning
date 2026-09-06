"""Build, validate, and store transition datasets without losing Episodes."""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from robot_learning.gymnasium_rollout import EpisodeResult


DATASET_SCHEMA_VERSION = 1
TRAIN_SPLIT_ID = 0
VALIDATION_SPLIT_ID = 1
TEST_SPLIT_ID = 2
VALID_SPLIT_IDS = (TRAIN_SPLIT_ID, VALIDATION_SPLIT_ID, TEST_SPLIT_ID)


@dataclass(frozen=True)
class CollectedEpisode:
    """One Episode plus metadata that remains constant along its trajectory."""

    episode_id: int
    environment_seed: int
    policy_id: int
    split_id: int
    result: EpisodeResult


@dataclass(frozen=True)
class TransitionDataset:
    """Flat transitions with IDs that preserve Episode and split boundaries."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_ids: np.ndarray
    step_ids: np.ndarray
    environment_seeds: np.ndarray
    policy_ids: np.ndarray
    split_ids: np.ndarray

    @property
    def transition_count(self) -> int:
        """Return the number of state-action transitions."""
        return int(self.rewards.shape[0])

    @property
    def episode_count(self) -> int:
        """Return the number of distinct Episodes."""
        return int(np.unique(self.episode_ids).size)


def assign_stratified_episode_splits(
    episode_ids: Sequence[int],
    policy_ids: Sequence[int],
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[int, int]:
    """Assign complete Episodes to splits separately within each policy group."""
    episode_array = np.asarray(episode_ids, dtype=np.int64)
    policy_array = np.asarray(policy_ids, dtype=np.int64)
    if episode_array.ndim != 1 or policy_array.shape != episode_array.shape:
        raise ValueError("episode_ids and policy_ids must be equal-length vectors")
    if episode_array.size == 0 or np.unique(episode_array).size != episode_array.size:
        raise ValueError("episode_ids must be non-empty and unique")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test split")

    rng = np.random.default_rng(seed)
    assignments: dict[int, int] = {}
    for policy_id in np.unique(policy_array):
        group_episode_ids = episode_array[policy_array == policy_id].copy()
        if group_episode_ids.size < 3:
            raise ValueError("each policy needs at least three Episodes")
        rng.shuffle(group_episode_ids)

        train_count = max(1, int(group_episode_ids.size * train_fraction))
        validation_count = max(
            1,
            int(group_episode_ids.size * validation_fraction),
        )
        if train_count + validation_count >= group_episode_ids.size:
            raise ValueError("split fractions leave no test Episode")

        for episode_id in group_episode_ids[:train_count]:
            assignments[int(episode_id)] = TRAIN_SPLIT_ID
        validation_end = train_count + validation_count
        for episode_id in group_episode_ids[train_count:validation_end]:
            assignments[int(episode_id)] = VALIDATION_SPLIT_ID
        for episode_id in group_episode_ids[validation_end:]:
            assignments[int(episode_id)] = TEST_SPLIT_ID
    return assignments


def build_transition_dataset(
    episodes: Sequence[CollectedEpisode],
) -> TransitionDataset:
    """Flatten complete Episodes while keeping all transition metadata."""
    if not episodes:
        raise ValueError("episodes must not be empty")
    episode_ids = [episode.episode_id for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode_id must be unique per CollectedEpisode")

    arrays: dict[str, list[np.ndarray]] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "terminated": [],
        "truncated": [],
        "episode_ids": [],
        "step_ids": [],
        "environment_seeds": [],
        "policy_ids": [],
        "split_ids": [],
    }
    for episode in episodes:
        result = episode.result
        step_count = result.step_count
        if step_count == 0:
            raise ValueError("each Episode must contain at least one transition")
        if result.observations.shape[0] != step_count + 1:
            raise ValueError("an Episode must contain T + 1 observations")
        if result.rewards.shape != (step_count,):
            raise ValueError("an Episode must contain one reward per action")
        if not (result.terminated or result.truncated):
            raise ValueError("an Episode must end by termination or truncation")

        # Row t connects observations[t] to observations[t + 1].
        arrays["observations"].append(result.observations[:-1])
        arrays["actions"].append(result.actions)
        arrays["rewards"].append(result.rewards)
        arrays["next_observations"].append(result.observations[1:])

        terminated = np.zeros(step_count, dtype=np.bool_)
        truncated = np.zeros(step_count, dtype=np.bool_)
        terminated[-1] = result.terminated
        truncated[-1] = result.truncated
        arrays["terminated"].append(terminated)
        arrays["truncated"].append(truncated)
        arrays["episode_ids"].append(
            np.full(step_count, episode.episode_id, dtype=np.int64)
        )
        arrays["step_ids"].append(np.arange(step_count, dtype=np.int64))
        arrays["environment_seeds"].append(
            np.full(step_count, episode.environment_seed, dtype=np.int64)
        )
        arrays["policy_ids"].append(
            np.full(step_count, episode.policy_id, dtype=np.int8)
        )
        arrays["split_ids"].append(
            np.full(step_count, episode.split_id, dtype=np.int8)
        )

    dataset = TransitionDataset(
        **{
            name: np.concatenate(parts, axis=0)
            for name, parts in arrays.items()
        }
    )
    validate_transition_dataset(dataset)
    return dataset


def validate_transition_dataset(
    dataset: TransitionDataset,
    *,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> None:
    """Reject shape, value, boundary, and Episode-continuity corruption."""
    transition_count = dataset.transition_count
    if transition_count == 0:
        raise ValueError("dataset must contain at least one transition")
    if dataset.observations.ndim != 2 or dataset.actions.ndim != 2:
        raise ValueError("observations and actions must be matrices")
    if dataset.next_observations.shape != dataset.observations.shape:
        raise ValueError("next_observations must match observations shape")

    for field in fields(dataset):
        array = getattr(dataset, field.name)
        if array.shape[0] != transition_count:
            raise ValueError(f"{field.name} has an inconsistent row count")
    for name in ("rewards", "terminated", "truncated"):
        if getattr(dataset, name).ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    for name in ("terminated", "truncated"):
        if getattr(dataset, name).dtype != np.bool_:
            raise ValueError(f"{name} must have boolean dtype")
    for name in ("observations", "actions", "rewards", "next_observations"):
        if not np.all(np.isfinite(getattr(dataset, name))):
            raise ValueError(f"{name} contains non-finite values")
    if not np.all(np.isin(dataset.split_ids, VALID_SPLIT_IDS)):
        raise ValueError("split_ids contains an unknown split")

    if (action_low is None) != (action_high is None):
        raise ValueError("action_low and action_high must be provided together")
    if action_low is not None and action_high is not None:
        low = np.asarray(action_low).reshape(1, -1)
        high = np.asarray(action_high).reshape(1, -1)
        if low.shape[1] != dataset.actions.shape[1] or high.shape != low.shape:
            raise ValueError("action bounds do not match action dimension")
        if np.any(dataset.actions < low) or np.any(dataset.actions > high):
            raise ValueError("actions exceed declared bounds")

    for episode_id in np.unique(dataset.episode_ids):
        rows = np.flatnonzero(dataset.episode_ids == episode_id)
        expected_steps = np.arange(rows.size, dtype=np.int64)
        if not np.array_equal(dataset.step_ids[rows], expected_steps):
            raise ValueError("step_ids must be contiguous inside each Episode")
        if np.any(dataset.terminated[rows[:-1]]) or np.any(
            dataset.truncated[rows[:-1]]
        ):
            raise ValueError("only the final transition may end an Episode")
        # TimeLimit may truncate on the same step that the task terminates.
        # Preserve both flags; learners need the original termination signal.
        if not (dataset.terminated[rows[-1]] or dataset.truncated[rows[-1]]):
            raise ValueError("final transition must terminate or truncate")
        if rows.size > 1 and not np.array_equal(
            dataset.next_observations[rows[:-1]],
            dataset.observations[rows[1:]],
        ):
            raise ValueError("next observation does not continue the Episode")
        for metadata in ("environment_seeds", "policy_ids", "split_ids"):
            if np.unique(getattr(dataset, metadata)[rows]).size != 1:
                raise ValueError(f"{metadata} changes inside one Episode")


def save_transition_dataset(
    path: str | Path,
    dataset: TransitionDataset,
) -> None:
    """Validate and save numeric arrays without Python pickle objects."""
    validate_transition_dataset(dataset)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int64),
        **{
            field.name: getattr(dataset, field.name)
            for field in fields(dataset)
        },
    )


def load_transition_dataset(path: str | Path) -> TransitionDataset:
    """Load an NPZ dataset with pickle disabled, then validate its contract."""
    with np.load(Path(path), allow_pickle=False) as archive:
        schema_version = int(archive["schema_version"])
        if schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(f"unsupported dataset schema: {schema_version}")
        dataset = TransitionDataset(
            **{
                field.name: archive[field.name].copy()
                for field in fields(TransitionDataset)
            }
        )
    validate_transition_dataset(dataset)
    return dataset
