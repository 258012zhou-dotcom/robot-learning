"""Compare one near-target speed reward change under the same PPO protocol."""

import csv
from dataclasses import asdict
import json
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
from robot_learning.ppo_training import (
    ACTION_VALUES,
    DiscreteActorCritic,
    collect_episodes,
    update_from_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "038_ppo_velocity_reward.json"
MODEL_PATH = PROJECT_ROOT / "experiments" / "017_mujoco_step" / "point_robot.xml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "038_ppo_velocity_reward"


class GreedyDiscretePolicy:
    """Use the most probable of the unchanged three discrete actions."""

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
    environment: PointRobotReachEnv, policy: Any, seeds: range
) -> tuple[dict[str, float | int], list[dict[str, float | int | bool]]]:
    episodes = [run_episode(environment, policy, seed=seed) for seed in seeds]
    records = [
        {
            "seed": seed,
            "target_position": float(episode.observations[0, 2]),
            "success": episode.is_success,
            "total_reward": episode.total_reward,
            "final_distance": abs(float(episode.observations[-1, 3])),
            "final_position": float(episode.observations[-1, 0]),
            "steps": episode.step_count,
        }
        for seed, episode in zip(seeds, episodes)
    ]
    return asdict(summarize_episodes(episodes)), records


def train_condition(
    base: dict[str, Any], *, penalty_weight: float, near_target_radius: float
) -> tuple[DiscreteActorCritic, list[dict[str, float | int]]]:
    """Train from the same initial seed and budget for either reward condition."""
    torch.manual_seed(int(base["seed"]))
    environment = make_environment(base)
    model = DiscreteActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(base["learning_rate"]))
    history: list[dict[str, float | int]] = []
    for update_index in range(int(base["training_updates"])):
        first_seed = int(base["seed"]) + update_index * int(base["episodes_per_update"])
        batch = collect_episodes(
            environment,
            model,
            episode_seeds=range(first_seed, first_seed + int(base["episodes_per_update"])),
            gamma=float(base["gamma"]),
            gae_lambda=float(base["gae_lambda"]),
            reward_scale=float(base["reward_scale"]),
            velocity_penalty_weight=penalty_weight,
            near_target_radius=near_target_radius,
        )
        metrics = update_from_batch(
            model,
            optimizer,
            batch,
            epochs=int(base["update_epochs"]),
            clip_epsilon=float(base["clip_epsilon"]),
            value_loss_weight=float(base["value_loss_weight"]),
            entropy_weight=float(base["entropy_weight"]),
        )
        history.append(
            {
                "update": update_index + 1,
                "steps_collected": int(batch.actions.numel()),
                "training_success_rate": sum(
                    bool(record["success"]) for record in batch.episode_records
                ) / len(batch.episode_records),
                "training_mean_raw_reward": float(np.mean([
                    float(record["total_reward"]) for record in batch.episode_records
                ])),
                "training_mean_speed_penalty": float(np.mean([
                    float(record["total_speed_penalty"]) for record in batch.episode_records
                ])),
                **metrics,
            }
        )
    environment.close()
    return model, history


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config: dict[str, Any] = json.load(file)
    base_path = PROJECT_ROOT / str(config["base_config"])
    with base_path.open(encoding="utf-8") as file:
        base: dict[str, Any] = json.load(file)
    penalty_weight = float(config["velocity_penalty_weight"])
    radius = float(config["near_target_radius"])
    if not np.isfinite(penalty_weight) or penalty_weight <= 0.0:
        raise ValueError("velocity_penalty_weight must be positive and finite")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("near_target_radius must be positive and finite")
    torch.set_num_threads(1)
    seeds = range(
        int(base["evaluation_seed"]),
        int(base["evaluation_seed"]) + int(base["evaluation_episode_count"]),
    )
    summaries: dict[str, dict[str, float | int]] = {}
    evaluation_records: dict[str, list[dict[str, float | int | bool]]] = {}
    training_history: dict[str, list[dict[str, float | int]]] = {}
    for name, weight in (("original_reward", 0.0), ("velocity_shaped", penalty_weight)):
        model, training_history[name] = train_condition(
            base, penalty_weight=weight, near_target_radius=radius
        )
        environment = make_environment(base)
        summaries[name], evaluation_records[name] = evaluate(
            environment, GreedyDiscretePolicy(model), seeds
        )
        environment.close()
        print(f"{name}: {summaries[name]}")

    environment = make_environment(base)
    summaries["random"], evaluation_records["random"] = evaluate(
        environment,
        RandomPolicy(environment.action_space, seed=seeds.start + 10000),
        seeds,
    )
    summaries["proportional"], evaluation_records["proportional"] = evaluate(
        environment,
        ProportionalReachPolicy(float(base["proportional_gain"]), environment.action_space),
        seeds,
    )
    environment.close()
    reference_targets = [
        record["target_position"] for record in evaluation_records["original_reward"]
    ]
    for name, records in evaluation_records.items():
        if [record["target_position"] for record in records] != reference_targets:
            raise RuntimeError(f"{name} did not receive the same evaluation targets")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "experiment_name": config["experiment_name"],
                "fixed_base_config": str(config["base_config"]),
                "base_settings": base,
                "experiment_settings": config,
                "only_training_change": {
                    "near_target_radius": radius,
                    "velocity_penalty_weight": penalty_weight,
                },
                "evaluation_seed_range": [seeds.start, seeds.stop - 1],
                "summaries": summaries,
                "training_history": training_history,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    with (OUTPUT_DIR / "evaluation_episodes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["condition", *evaluation_records["random"][0]])
        writer.writeheader()
        for name, records in evaluation_records.items():
            for record in records:
                writer.writerow({"condition": name, **record})


if __name__ == "__main__":
    main()
