"""Compare greedy and sampled evaluation of the same saved PPO actors."""

import csv
import json
import math

import numpy as np
import torch

from robot_learning.gymnasium_rollout import EpisodeResult, run_episode
from robot_learning.ppo_action_selection import SampledDiscretePolicy
from robot_learning.ppo_training import DiscreteActorCritic

from analyze_failures import check_replay
from run import CONFIG_PATH, OUTPUT_DIR, GreedyDiscretePolicy, make_environment


def record_episode(
    episode: EpisodeResult,
    *,
    training_seed: int,
    mode: str,
    sampling_seed: int | str,
    evaluation_seed: int,
) -> dict[str, float | int | str | bool]:
    """Keep start and full-Episode outcomes separate."""
    last = episode.observations[-1]
    return {
        "training_seed": training_seed,
        "mode": mode,
        "sampling_seed": sampling_seed,
        "evaluation_seed": evaluation_seed,
        "target": float(episode.observations[0, 2]),
        "first_action": float(episode.actions[0, 0]),
        "ever_nonzero_action": bool(np.any(episode.actions[:, 0] != 0.0)),
        "success": episode.is_success,
        "steps": episode.step_count,
        "total_reward": episode.total_reward,
        "final_distance": abs(float(last[3])),
        "final_position": float(last[0]),
        "final_velocity": float(last[1]),
    }


def summarize(rows: list[dict[str, float | int | str | bool]]) -> dict[str, float | int]:
    """Summarize one model under one action-selection run."""
    return {
        "episode_count": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "positive_success_count": sum(
            bool(row["success"]) for row in rows if float(row["target"]) > 0
        ),
        "negative_success_count": sum(
            bool(row["success"]) for row in rows if float(row["target"]) < 0
        ),
        "started_count": sum(bool(row["ever_nonzero_action"]) for row in rows),
        "mean_total_reward": float(np.mean([float(row["total_reward"]) for row in rows])),
        "mean_final_distance": float(np.mean([float(row["final_distance"]) for row in rows])),
    }


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        experiment = json.load(file)
    with (CONFIG_PATH.parents[1] / experiment["base_config"]).open(encoding="utf-8") as file:
        base = json.load(file)
    sampling_seeds = [int(seed) for seed in experiment["sampling_seeds"]]
    if not sampling_seeds or len(sampling_seeds) != len(set(sampling_seeds)) or min(sampling_seeds) < 0:
        raise ValueError("sampling seeds must be distinct non-negative integers")
    evaluation_seeds = range(
        int(base["evaluation_seed"]),
        int(base["evaluation_seed"]) + int(base["evaluation_episode_count"]),
    )
    with (OUTPUT_DIR / "evaluation_episodes.csv").open(encoding="utf-8", newline="") as file:
        reference = {
            (row["policy"], row["training_seed"], int(row["seed"])): row
            for row in csv.DictReader(file)
        }

    torch.set_num_threads(1)
    environment = make_environment(base)
    all_rows: list[dict[str, float | int | str | bool]] = []
    per_training_seed = {}
    try:
        for training_seed in experiment["training_seeds"]:
            model = DiscreteActorCritic()
            model.load_state_dict(torch.load(
                OUTPUT_DIR / f"trained_seed_{training_seed}.pt",
                map_location="cpu", weights_only=True,
            ))
            model.eval()
            greedy_policy = GreedyDiscretePolicy(model)
            greedy_rows = []
            for evaluation_seed in evaluation_seeds:
                episode = run_episode(environment, greedy_policy, seed=evaluation_seed)
                check_replay(
                    episode,
                    policy_name=f"ppo_{training_seed}",
                    seed=evaluation_seed,
                    reference=reference,
                )
                greedy_rows.append(record_episode(
                    episode,
                    training_seed=training_seed,
                    mode="greedy",
                    sampling_seed="",
                    evaluation_seed=evaluation_seed,
                ))
            all_rows.extend(greedy_rows)

            repeats = []
            for sampling_seed in sampling_seeds:
                sampled_rows = []
                for offset, evaluation_seed in enumerate(evaluation_seeds):
                    # Isolate each task's random draws from other Episode lengths.
                    episode_rng_seed = (
                        sampling_seed * 1_000_000 + training_seed * 1_000 + offset
                    )
                    policy = SampledDiscretePolicy(model, seed=episode_rng_seed)
                    episode = run_episode(environment, policy, seed=evaluation_seed)
                    row = record_episode(
                        episode,
                        training_seed=training_seed,
                        mode="sampled",
                        sampling_seed=sampling_seed,
                        evaluation_seed=evaluation_seed,
                    )
                    greedy_target = greedy_rows[offset]["target"]
                    if not math.isclose(float(row["target"]), float(greedy_target), abs_tol=1e-6):
                        raise RuntimeError("sampled and greedy runs received different tasks")
                    sampled_rows.append(row)
                all_rows.extend(sampled_rows)
                repeats.append({"sampling_seed": sampling_seed, **summarize(sampled_rows)})
            per_training_seed[str(training_seed)] = {
                "greedy": summarize(greedy_rows),
                "sampled_repeats": repeats,
                "sampled_success_count_range": [
                    min(int(repeat["success_count"]) for repeat in repeats),
                    max(int(repeat["success_count"]) for repeat in repeats),
                ],
                "sampled_mean_success_count": float(np.mean([
                    int(repeat["success_count"]) for repeat in repeats
                ])),
            }
    finally:
        environment.close()

    result = {
        "comparison": "saved actors: greedy argmax vs sampled categorical actions",
        "training_seeds": experiment["training_seeds"],
        "evaluation_seeds": [evaluation_seeds.start, evaluation_seeds.stop - 1],
        "sampling_seeds": sampling_seeds,
        "episode_rng_seed_formula": "sampling_seed * 1000000 + training_seed * 1000 + evaluation_offset",
        "greedy_replay_verified": True,
        "per_training_seed": per_training_seed,
    }
    with (OUTPUT_DIR / "action_selection_results.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "action_selection_episodes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    for training_seed, outcome in per_training_seed.items():
        print(
            f"train={training_seed}: greedy="
            f"{outcome['greedy']['success_count']}/{len(evaluation_seeds)}, "
            f"sampled={[item['success_count'] for item in outcome['sampled_repeats']]}, "
            f"sampled_started={[item['started_count'] for item in outcome['sampled_repeats']]}"
        )


if __name__ == "__main__":
    main()
