"""Repeat the unchanged experiment-037 PPO training with independent seeds."""

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_learning.gymnasium_rollout import (
    ProportionalReachPolicy,
    RandomPolicy,
    run_episode,
    summarize_episodes,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.ppo_multiseed import (
    summarize_success_variation,
    training_seed_ranges,
)
from robot_learning.ppo_training import (
    ACTION_VALUES,
    DiscreteActorCritic,
    collect_episodes,
    update_from_batch,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "039_ppo_multiseed.json"
MODEL_PATH = ROOT / "experiments" / "017_mujoco_step" / "point_robot.xml"
OUTPUT_DIR = ROOT / "outputs" / "039_ppo_multiseed"
REFERENCE_CSV = ROOT / "outputs" / "037_ppo_point_robot" / "evaluation_episodes.csv"


class GreedyDiscretePolicy:
    """Choose the largest actor logit, exactly as in experiment 037."""

    def __init__(self, model: DiscreteActorCritic) -> None:
        self.model = model

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self.model.actor(torch.as_tensor(observation, dtype=torch.float32))
            action_index = int(logits.argmax())
        return np.asarray([ACTION_VALUES[action_index]], dtype=np.float32)


def make_environment(config: dict[str, Any]) -> PointRobotReachEnv:
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


def evaluate(
    environment: PointRobotReachEnv, policy: Any, seeds: range, config: dict[str, Any]
) -> tuple[dict[str, float | int], list[dict[str, float | int | bool]]]:
    episodes = [run_episode(environment, policy, seed=seed) for seed in seeds]
    records = []
    for seed, episode in zip(seeds, episodes):
        target = float(episode.observations[0, 2])
        final_position = float(episode.observations[-1, 0])
        final_velocity = float(episode.observations[-1, 1])
        final_distance = abs(float(episode.observations[-1, 3]))
        records.append({
            "seed": seed,
            "target_position": target,
            "success": episode.is_success,
            "terminated": episode.terminated,
            "truncated": episode.truncated,
            "steps": episode.step_count,
            "total_reward": episode.total_reward,
            "final_distance": final_distance,
            "final_position": final_position,
            "final_velocity": final_velocity,
            "mean_action": float(episode.actions.mean()),
            "end_position_within_tolerance": final_distance <= float(config["success_tolerance"]),
            "end_speed_within_tolerance": abs(final_velocity) <= float(config["velocity_tolerance"]),
            "end_beyond_target": target * (final_position - target) > 0.0,
        })
    summary = asdict(summarize_episodes(episodes))
    summary["positive_success_count"] = sum(
        bool(record["success"]) for record in records if record["target_position"] > 0
    )
    summary["negative_success_count"] = sum(
        bool(record["success"]) for record in records if record["target_position"] < 0
    )
    return summary, records


def train_one_seed(
    config: dict[str, Any], training_seeds: range, evaluation_seeds: range
) -> tuple[DiscreteActorCritic, dict[str, dict[str, float | int]],
           dict[str, list[dict[str, float | int | bool]]], list[dict[str, float | int]]]:
    """Match experiment 037's initialization, initial evaluation, and updates."""
    torch.manual_seed(training_seeds.start)
    environment = make_environment(config)
    if not np.array_equal(environment.action_space.low, [-1.0]) or not np.array_equal(
        environment.action_space.high, [1.0]
    ):
        raise ValueError("the discrete action set requires environment bounds [-1, 1]")
    model = DiscreteActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    summaries: dict[str, dict[str, float | int]] = {}
    records: dict[str, list[dict[str, float | int | bool]]] = {}
    summaries["initial_greedy"], records["initial_greedy"] = evaluate(
        environment, GreedyDiscretePolicy(model), evaluation_seeds, config
    )
    history: list[dict[str, float | int]] = []
    for update_index in range(int(config["training_updates"])):
        first_seed = training_seeds.start + update_index * int(config["episodes_per_update"])
        batch = collect_episodes(
            environment,
            model,
            episode_seeds=range(first_seed, first_seed + int(config["episodes_per_update"])),
            gamma=float(config["gamma"]),
            gae_lambda=float(config["gae_lambda"]),
            reward_scale=float(config["reward_scale"]),
        )
        metrics = update_from_batch(
            model,
            optimizer,
            batch,
            epochs=int(config["update_epochs"]),
            clip_epsilon=float(config["clip_epsilon"]),
            value_loss_weight=float(config["value_loss_weight"]),
            entropy_weight=float(config["entropy_weight"]),
        )
        history.append({
            "update": update_index + 1,
            "steps_collected": int(batch.actions.numel()),
            "training_success_rate": sum(
                bool(record["success"]) for record in batch.episode_records
            ) / len(batch.episode_records),
            "training_mean_reward": float(np.mean([
                float(record["total_reward"]) for record in batch.episode_records
            ])),
            **metrics,
        })
    summaries["trained_greedy"], records["trained_greedy"] = evaluate(
        environment, GreedyDiscretePolicy(model), evaluation_seeds, config
    )
    environment.close()
    return model, summaries, records, history


def compare_with_experiment_037(
    current_records: list[dict[str, float | int | bool]],
) -> dict[str, Any]:
    """Compare seed 37 per Episode when the ignored old output is available."""
    if not REFERENCE_CSV.exists():
        return {"status": "reference_output_unavailable"}
    with REFERENCE_CSV.open(encoding="utf-8", newline="") as file:
        old_records = [
            row for row in csv.DictReader(file) if row["policy"] == "trained_greedy"
        ]
    old_by_seed = {int(row["seed"]): row for row in old_records}
    mismatch_seeds = []
    for current in current_records:
        seed = int(current["seed"])
        old = old_by_seed.get(seed)
        if old is None or old["success"] != str(current["success"]) or int(old["steps"]) != current["steps"]:
            mismatch_seeds.append(seed)
            continue
        if any(not math.isclose(
            float(old[key]), float(current[key]), rel_tol=1e-8, abs_tol=1e-6
        ) for key in ("target_position", "total_reward", "final_distance", "final_position", "final_velocity")):
            mismatch_seeds.append(seed)
    return {
        "status": "matched" if not mismatch_seeds and len(old_records) == len(current_records) else "mismatch",
        "mismatch_seeds": mismatch_seeds,
        "reference_episode_count": len(old_records),
    }


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        experiment: dict[str, Any] = json.load(file)
    with (ROOT / str(experiment["base_config"])).open(encoding="utf-8") as file:
        base: dict[str, Any] = json.load(file)
    starts = [int(seed) for seed in experiment["training_seeds"]]
    evaluation_seeds = range(
        int(base["evaluation_seed"]),
        int(base["evaluation_seed"]) + int(base["evaluation_episode_count"]),
    )
    ranges = training_seed_ranges(
        starts,
        int(base["training_updates"]) * int(base["episodes_per_update"]),
        evaluation_seeds,
    )
    torch.set_num_threads(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_summaries: dict[str, Any] = {}
    run_histories: dict[str, Any] = {}
    evaluation_rows: list[dict[str, Any]] = []
    target_reference: list[float] | None = None
    reference_check: dict[str, Any] = {"status": "not_checked"}
    for training_range in ranges:
        seed = training_range.start
        model, summaries, records, history = train_one_seed(
            base, training_range, evaluation_seeds
        )
        trained_records = records["trained_greedy"]
        targets = [float(record["target_position"]) for record in trained_records]
        if target_reference is None:
            target_reference = targets
        elif targets != target_reference:
            raise RuntimeError("training runs received different evaluation tasks")
        if seed == int(base["seed"]):
            reference_check = compare_with_experiment_037(trained_records)
        run_summaries[str(seed)] = summaries
        run_histories[str(seed)] = history
        for policy_name, policy_records in records.items():
            evaluation_rows.extend(
                {"training_seed": seed, "policy": policy_name, **record}
                for record in policy_records
            )
        torch.save(model.state_dict(), OUTPUT_DIR / f"trained_seed_{seed}.pt")
        print(f"training_seed={seed}: {summaries['trained_greedy']}", flush=True)

    environment = make_environment(base)
    baselines: dict[str, dict[str, float | int]] = {}
    for name, policy in (
        ("random", RandomPolicy(environment.action_space, seed=evaluation_seeds.start + 10000)),
        ("proportional", ProportionalReachPolicy(float(base["proportional_gain"]), environment.action_space)),
    ):
        baselines[name], records = evaluate(environment, policy, evaluation_seeds, base)
        if [float(record["target_position"]) for record in records] != target_reference:
            raise RuntimeError(f"{name} received different evaluation tasks")
        evaluation_rows.extend(
            {"training_seed": "", "policy": name, **record} for record in records
        )
    environment.close()

    success_counts = [
        int(run_summaries[str(seed)]["trained_greedy"]["success_count"]) for seed in starts
    ]
    result = {
        "experiment_name": experiment["experiment_name"],
        "base_config": str(experiment["base_config"]),
        "base_settings": base,
        "training_seed_ranges": [[items.start, items.stop - 1] for items in ranges],
        "evaluation_seed_range": [evaluation_seeds.start, evaluation_seeds.stop - 1],
        "evaluation_mode": "greedy actor for learned policies",
        "per_training_seed": run_summaries,
        "training_history": run_histories,
        "success_variation": summarize_success_variation(success_counts, len(evaluation_seeds)),
        "baselines": baselines,
        "experiment_037_reference_check": reference_check,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "evaluation_episodes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(evaluation_rows[0]))
        writer.writeheader()
        writer.writerows(evaluation_rows)
    print(json.dumps({
        "success_variation": result["success_variation"],
        "baselines": baselines,
        "experiment_037_reference_check": reference_check,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
