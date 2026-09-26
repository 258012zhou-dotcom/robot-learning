"""Compare 200- and 400-step evaluations without changing any policy."""

from collections import Counter
import csv
import json

import numpy as np
import torch

from robot_learning.gymnasium_rollout import ProportionalReachPolicy, run_episode
from robot_learning.reach_trajectory_analysis import summarize_reach_trajectory
from robot_learning.sac_evaluation import DeterministicSACPolicy

from analyze_multiseed_failures import classify
from analyze_trajectories import check_episode, reproduce_actor
from run import ROOT, OUTPUT_DIR, make_environment
from run_multiseed import CONFIG_PATH


EXTENDED_EPISODE_STEPS = 400


def compare_prefix(short, long) -> None:
    """Ensure the horizon change did not alter any action or physical state."""
    steps = short.step_count
    if long.step_count < steps:
        raise RuntimeError("the longer horizon ended before the original trajectory")
    for name, before, after in (
        ("observation", short.observations, long.observations[:steps + 1]),
        ("action", short.actions, long.actions[:steps]),
        ("reward", short.rewards, long.rewards[:steps]),
    ):
        if not np.array_equal(before, after):
            raise RuntimeError(f"the first {steps} steps changed: {name}")
    if short.is_success and (not long.is_success or long.step_count != steps):
        raise RuntimeError("a previously successful trajectory changed")
    if not short.is_success and not short.truncated:
        raise RuntimeError("the original unsuccessful trajectory was not time-limited")


def evaluate_pair(config: dict, policy, saved: dict, *, name: str, training_seed: int | None,
                  evaluation_seeds: range) -> list[dict]:
    short_env = make_environment(config, max_episode_steps=int(config["evaluation_episode_steps"]))
    long_env = make_environment(config, max_episode_steps=EXTENDED_EPISODE_STEPS)
    rows = []
    try:
        for seed in evaluation_seeds:
            short = run_episode(short_env, policy, seed=seed)
            check_episode(short, saved[(training_seed, name, seed)], policy=name, seed=seed)
            long = run_episode(long_env, policy, seed=seed)
            compare_prefix(short, long)
            short_diagnostic = summarize_reach_trajectory(
                short, position_tolerance=short_env.success_tolerance,
                velocity_tolerance=short_env.velocity_tolerance,
            )
            long_diagnostic = summarize_reach_trajectory(
                long, position_tolerance=long_env.success_tolerance,
                velocity_tolerance=long_env.velocity_tolerance,
            )
            rows.append({
                "training_seed": training_seed,
                "policy": name,
                "evaluation_seed": seed,
                "target": float(short.observations[0, 2]),
                "success_200": short.is_success,
                "success_400": long.is_success,
                "steps_200": short.step_count,
                "steps_400": long.step_count,
                "category_200": classify(short.is_success, short_diagnostic),
                "category_400": classify(long.is_success, long_diagnostic),
                "first_near_200": short_diagnostic["first_near_step"],
                "first_near_400": long_diagnostic["first_near_step"],
                "final_distance_200": abs(float(short.observations[-1, 3])),
                "final_distance_400": abs(float(long.observations[-1, 3])),
                "final_speed_200": abs(float(short.observations[-1, 1])),
                "final_speed_400": abs(float(long.observations[-1, 1])),
            })
    finally:
        short_env.close()
        long_env.close()
    return rows


def summarize(rows: list[dict]) -> dict:
    counts_200 = Counter(row["category_200"] for row in rows)
    counts_400 = Counter(row["category_400"] for row in rows)
    return {
        "training_seed": rows[0]["training_seed"],
        "policy": rows[0]["policy"],
        "episodes": len(rows),
        "success_200": counts_200["success"],
        "success_400": counts_400["success"],
        "late_success": sum(not row["success_200"] and row["success_400"] for row in rows),
        "never_near_200": counts_200["never_near_position"],
        "never_near_400": counts_400["never_near_position"],
        "near_but_fast_200": counts_200["near_position_but_never_slow"],
        "near_but_fast_400": counts_400["near_position_but_never_slow"],
        "positive_success_200": sum(row["success_200"] and row["target"] > 0 for row in rows),
        "positive_success_400": sum(row["success_400"] and row["target"] > 0 for row in rows),
        "negative_success_200": sum(row["success_200"] and row["target"] < 0 for row in rows),
        "negative_success_400": sum(row["success_400"] and row["target"] < 0 for row in rows),
    }


def main() -> None:
    with (OUTPUT_DIR / "multiseed_results.json").open(encoding="utf-8") as file:
        original = json.load(file)
    with CONFIG_PATH.open(encoding="utf-8") as file:
        extension = json.load(file)
    with (ROOT / extension["base_config"]).open(encoding="utf-8") as file:
        base = json.load(file)
    if original["multiseed_protocol"] != extension or original["base_protocol"] != base:
        raise RuntimeError("the saved evaluation protocol changed")
    if int(base["evaluation_episode_steps"]) != 200:
        raise RuntimeError("the original evaluation was not 200 steps")
    evaluation_seeds = range(
        int(extension["evaluation_seed"]),
        int(extension["evaluation_seed"]) + int(extension["evaluation_episode_count"]),
    )
    with (OUTPUT_DIR / "multiseed_episodes.csv").open(encoding="utf-8", newline="") as file:
        saved = {
            (int(row["training_seed"]) if row["training_seed"] else None,
             row["policy"], int(row["seed"])): row
            for row in csv.DictReader(file)
        }
    expected = (2 * len(extension["training_seeds"]) + 1) * len(evaluation_seeds)
    if len(saved) != expected:
        raise RuntimeError("original episode records are incomplete")

    torch.set_num_threads(1)
    rows = []
    for item in original["per_training_seed"]:
        training_seed = int(item["training_seed"])
        config = {**base, "training_seed": training_seed}
        for name in ("positive_only", "balanced"):
            actor = reproduce_actor(
                config, balanced=(name == "balanced"), reference=item[name]["training"],
            )
            group = evaluate_pair(
                config, DeterministicSACPolicy(actor), saved,
                name=name, training_seed=training_seed, evaluation_seeds=evaluation_seeds,
            )
            rows.extend(group)
            print(f"verified {training_seed} {name}: {summarize(group)['success_200']} -> "
                  f"{summarize(group)['success_400']}/{len(group)}", flush=True)

    environment = make_environment(base, max_episode_steps=int(base["evaluation_episode_steps"]))
    try:
        proportional = ProportionalReachPolicy(float(base["proportional_gain"]), environment.action_space)
    finally:
        environment.close()
    group = evaluate_pair(
        base, proportional, saved, name="proportional", training_seed=None,
        evaluation_seeds=evaluation_seeds,
    )
    rows.extend(group)
    counts = [summarize([row for row in rows if row["training_seed"] == seed and
                         row["policy"] == name])
              for seed in extension["training_seeds"] for name in ("positive_only", "balanced")]
    counts.append(summarize(group))
    report = {
        "replay_verified": True,
        "prefix_exactly_equal": True,
        "original_episode_steps": 200,
        "extended_episode_steps": EXTENDED_EPISODE_STEPS,
        "training_unchanged": True,
        "task_seeds": list(evaluation_seeds),
        "counts": counts,
    }
    with (OUTPUT_DIR / "horizon_results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "horizon_episodes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
