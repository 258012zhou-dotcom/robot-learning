"""Separate position and speed failures across all six SAC training runs."""

from collections import Counter
import csv
import json

import numpy as np
import torch

from robot_learning.gymnasium_rollout import run_episode
from robot_learning.reach_trajectory_analysis import summarize_reach_trajectory
from robot_learning.sac_evaluation import DeterministicSACPolicy

from analyze_trajectories import check_episode, reproduce_actor, trace_rows
from run import OUTPUT_DIR, make_environment
from run_multiseed import CONFIG_PATH


CASE_KEYS = {
    (126, "balanced", 300033),  # Positive-target success.
    (226, "balanced", 300027),  # Near target, but still too fast.
    (226, "positive_only", 300027),  # Same task, other training arm.
}


def classify(success: bool, diagnostics: dict) -> str:
    """Use complete trajectories, not only endpoint distance."""
    reached_and_slow = diagnostics["near_and_slow_steps"] > 0
    if success != reached_and_slow:
        raise RuntimeError("success flag disagrees with position-and-speed condition")
    if success:
        return "success"
    if diagnostics["first_near_step"] is None:
        return "never_near_position"
    return "near_position_but_never_slow"


def main() -> None:
    with (OUTPUT_DIR / "multiseed_results.json").open(encoding="utf-8") as file:
        original = json.load(file)
    with CONFIG_PATH.open(encoding="utf-8") as file:
        extension = json.load(file)
    if original["multiseed_protocol"] != extension:
        raise RuntimeError("multiseed protocol has changed since evaluation")
    base = original["base_protocol"]
    seeds = range(
        int(extension["evaluation_seed"]),
        int(extension["evaluation_seed"]) + int(extension["evaluation_episode_count"]),
    )
    with (OUTPUT_DIR / "multiseed_episodes.csv").open(encoding="utf-8", newline="") as file:
        saved = {
            (int(row["training_seed"]), row["policy"], int(row["seed"])): row
            for row in csv.DictReader(file) if row["policy"] != "proportional"
        }
    if len(saved) != 2 * len(seeds) * len(extension["training_seeds"]):
        raise RuntimeError("saved evaluation rows are incomplete")

    torch.set_num_threads(1)
    episodes: list[dict] = []
    selected_steps: list[dict] = []
    for item in original["per_training_seed"]:
        training_seed = int(item["training_seed"])
        config = {**base, "training_seed": training_seed}
        for policy in ("positive_only", "balanced"):
            actor = reproduce_actor(
                config, balanced=(policy == "balanced"),
                reference=item[policy]["training"],
            )
            environment = make_environment(
                config, max_episode_steps=int(config["evaluation_episode_steps"]),
            )
            try:
                for seed in seeds:
                    episode = run_episode(environment, DeterministicSACPolicy(actor), seed=seed)
                    check_episode(
                        episode, saved[(training_seed, policy, seed)],
                        policy=policy, seed=seed,
                    )
                    diagnostics = summarize_reach_trajectory(
                        episode,
                        position_tolerance=environment.success_tolerance,
                        velocity_tolerance=environment.velocity_tolerance,
                    )
                    category = classify(episode.is_success, diagnostics)
                    episodes.append({
                        "training_seed": training_seed,
                        "policy": policy,
                        "evaluation_seed": seed,
                        "target": float(episode.observations[0, 2]),
                        "success": episode.is_success,
                        "category": category,
                        "steps": episode.step_count,
                        "closest_distance": float(np.min(np.abs(episode.observations[:, 3]))),
                        "final_distance": abs(float(episode.observations[-1, 3])),
                        "final_speed": abs(float(episode.observations[-1, 1])),
                        **diagnostics,
                    })
                    if (training_seed, policy, seed) in CASE_KEYS:
                        selected_steps.extend({"training_seed": training_seed, **row} for row in
                                              trace_rows(episode, policy=policy, seed=seed))
            finally:
                environment.close()
            print(f"verified seed {training_seed} {policy}: {len(seeds)} episodes", flush=True)

    counts = []
    for training_seed in extension["training_seeds"]:
        for policy in ("positive_only", "balanced"):
            group = [row for row in episodes if row["training_seed"] == training_seed and row["policy"] == policy]
            categories = Counter(row["category"] for row in group)
            counts.append({
                "training_seed": training_seed,
                "policy": policy,
                "episodes": len(group),
                "success": categories["success"],
                "never_near_position": categories["never_near_position"],
                "near_position_but_never_slow": categories["near_position_but_never_slow"],
                "crossed_target": sum(row["first_crossing_step"] is not None for row in group),
                "positive_success": sum(row["success"] and row["target"] > 0 for row in group),
                "negative_success": sum(row["success"] and row["target"] < 0 for row in group),
            })
            if sum(categories.values()) != len(seeds):
                raise RuntimeError("failure categories did not cover every evaluation episode")

    if len(selected_steps) == 0 or len(episodes) != len(saved):
        raise RuntimeError("diagnostic records are incomplete")
    report = {
        "replay_verified": True,
        "categories": {
            "success": "position and speed simultaneously within tolerance",
            "never_near_position": "position never within tolerance",
            "near_position_but_never_slow": "position within tolerance, speed never simultaneously low enough",
        },
        "case_keys": [list(key) for key in sorted(CASE_KEYS)],
        "counts": counts,
        "episodes": episodes,
    }
    with (OUTPUT_DIR / "multiseed_failure_modes.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "multiseed_failure_steps.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(selected_steps[0]))
        writer.writeheader()
        writer.writerows(selected_steps)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
