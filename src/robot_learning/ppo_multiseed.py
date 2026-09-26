"""Small, explicit checks for a controlled multi-seed PPO evaluation."""

from collections.abc import Sequence


def training_seed_ranges(
    starts: Sequence[int], episodes_per_run: int, evaluation_seeds: range
) -> list[range]:
    """Keep training streams separate from each other and the held-out tasks."""
    if len(starts) < 2 or episodes_per_run <= 0 or not evaluation_seeds:
        raise ValueError("need multiple training seeds and nonempty positive ranges")
    if any(seed < 0 for seed in starts) or len(set(starts)) != len(starts):
        raise ValueError("training seeds must be distinct non-negative integers")
    ranges = [range(seed, seed + episodes_per_run) for seed in starts]
    for left_index, left in enumerate(ranges):
        if set(left).intersection(evaluation_seeds):
            raise ValueError("training and evaluation seeds overlap")
        for right in ranges[left_index + 1:]:
            if left.start < right.stop and right.start < left.stop:
                raise ValueError("training seed ranges overlap")
    return ranges


def summarize_success_variation(
    success_counts: Sequence[int], episodes_per_run: int
) -> dict[str, int | float | list[int]]:
    """Report every run and its spread, never only the best trained policy."""
    if len(success_counts) < 2 or episodes_per_run <= 0:
        raise ValueError("need multiple runs and a positive evaluation count")
    if any(count < 0 or count > episodes_per_run for count in success_counts):
        raise ValueError("success counts must be within the evaluation count")
    return {
        "training_run_count": len(success_counts),
        "evaluation_episodes_per_run": episodes_per_run,
        "success_counts": list(success_counts),
        "minimum_success_count": min(success_counts),
        "maximum_success_count": max(success_counts),
        "mean_success_rate": sum(success_counts) / (len(success_counts) * episodes_per_run),
    }
