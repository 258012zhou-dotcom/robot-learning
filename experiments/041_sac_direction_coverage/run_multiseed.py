"""Repeat experiment 041 as three paired training seeds on fresh held-out tasks."""

import csv
from dataclasses import asdict
import json

import torch

from robot_learning.gymnasium_rollout import ProportionalReachPolicy
from robot_learning.sac_evaluation import evaluate_reach_policy

from run import ROOT, OUTPUT_DIR, make_environment, train_arm


CONFIG_PATH = ROOT / "configs/041_sac_direction_multiseed.json"


def directional_success(records: list[dict]) -> dict[str, int]:
    """Keep positive and negative task denominators visible."""
    return {
        "positive_tasks": sum(record["target_position"] > 0 for record in records),
        "positive_success": sum(
            record["target_position"] > 0 and record["success"] for record in records
        ),
        "negative_tasks": sum(record["target_position"] < 0 for record in records),
        "negative_success": sum(
            record["target_position"] < 0 and record["success"] for record in records
        ),
    }


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        extension = json.load(file)
    with (ROOT / extension["base_config"]).open(encoding="utf-8") as file:
        base_config = json.load(file)
    training_seeds = [int(seed) for seed in extension["training_seeds"]]
    if len(training_seeds) != 3 or len(set(training_seeds)) != 3:
        raise ValueError("exactly three distinct training seeds are required")
    evaluation_start = int(extension["evaluation_seed"])
    evaluation_count = int(extension["evaluation_episode_count"])
    if evaluation_count <= 0:
        raise ValueError("evaluation count must be positive")
    evaluation_seeds = range(evaluation_start, evaluation_start + evaluation_count)
    for seed in training_seeds:
        possible_training_seeds = range(seed, seed + int(base_config["training_steps"]) + 1)
        if set(evaluation_seeds).intersection(possible_training_seeds):
            raise ValueError("training and evaluation seeds overlap")

    torch.set_num_threads(1)
    per_seed: list[dict] = []
    episode_rows: list[dict] = []
    for seed in training_seeds:
        config = {**base_config, "training_seed": seed, "evaluation_seed": evaluation_start,
                  "evaluation_episode_count": evaluation_count}
        positive_training, positive_evaluation, positive_records, initial_parameters = train_arm(
            config, alternate_direction=False, evaluation_seeds=evaluation_seeds,
            initial_parameters=None,
        )
        balanced_training, balanced_evaluation, balanced_records, _ = train_arm(
            config, alternate_direction=True, evaluation_seeds=evaluation_seeds,
            initial_parameters=initial_parameters,
        )
        for positive, balanced in zip(positive_records, balanced_records, strict=True):
            if positive["seed"] != balanced["seed"] or positive["target_position"] != balanced["target_position"]:
                raise RuntimeError("paired arms did not receive the same evaluation task")
        positive_directions = directional_success(positive_records)
        balanced_directions = directional_success(balanced_records)
        if (positive_directions["positive_tasks"], positive_directions["negative_tasks"]) != (
            balanced_directions["positive_tasks"], balanced_directions["negative_tasks"]
        ):
            raise RuntimeError("paired evaluation directions differ")
        per_seed.append({
            "training_seed": seed,
            "positive_only": {
                **positive_training,
                "evaluation": positive_evaluation,
                "directional_success": positive_directions,
            },
            "balanced": {
                **balanced_training,
                "evaluation": balanced_evaluation,
                "directional_success": balanced_directions,
            },
            "paired_success_difference": (
                balanced_evaluation["success_count"] - positive_evaluation["success_count"]
            ),
        })
        for policy_name, records in (("positive_only", positive_records), ("balanced", balanced_records)):
            episode_rows.extend({"training_seed": seed, "policy": policy_name, **record} for record in records)
        print(
            f"seed {seed}: positive-only {positive_evaluation['success_count']}/{evaluation_count}, "
            f"balanced {balanced_evaluation['success_count']}/{evaluation_count}",
            flush=True,
        )

    # P is a fixed controller, so run it once on the same fresh task set.
    environment = make_environment(
        base_config, max_episode_steps=int(base_config["evaluation_episode_steps"]),
    )
    try:
        proportional_evaluation, proportional_records = evaluate_reach_policy(
            environment,
            ProportionalReachPolicy(float(base_config["proportional_gain"]), environment.action_space),
            evaluation_seeds,
        )
    finally:
        environment.close()
    task_targets = {record["seed"]: record["target_position"] for record in proportional_records}
    if any(record["target_position"] != task_targets[record["seed"]] for record in episode_rows):
        raise RuntimeError("trained policies and P controller saw different tasks")
    episode_rows.extend({"training_seed": "", "policy": "proportional", **record} for record in proportional_records)

    report = {
        "base_protocol": base_config,
        "multiseed_protocol": extension,
        "per_training_seed": per_seed,
        "proportional": {
            "evaluation": asdict(proportional_evaluation),
            "directional_success": directional_success(proportional_records),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "multiseed_results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "multiseed_episodes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    print(json.dumps({
        "per_training_seed": [
            {"training_seed": item["training_seed"],
             "positive_only": item["positive_only"]["evaluation"]["success_count"],
             "balanced": item["balanced"]["evaluation"]["success_count"],
             "paired_success_difference": item["paired_success_difference"]}
            for item in per_seed
        ],
        "proportional": asdict(proportional_evaluation),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
