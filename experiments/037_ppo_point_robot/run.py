"""Train a small discrete-action PPO policy and evaluate it on held-out tasks."""

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
CONFIG_PATH = PROJECT_ROOT / "configs" / "037_ppo_point_robot.json"
MODEL_PATH = PROJECT_ROOT / "experiments" / "017_mujoco_step" / "point_robot.xml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "037_ppo_point_robot"


class GreedyDiscretePolicy:
    """Evaluate a trained actor without adding action-sampling variance."""

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
    environment: PointRobotReachEnv,
    policy: Any,
    seeds: range,
) -> tuple[dict[str, float | int], list[dict[str, float | int | bool]]]:
    results = [run_episode(environment, policy, seed=seed) for seed in seeds]
    records = [
        {
            "seed": seed,
            "target_position": float(result.observations[0, 2]),
            "success": result.is_success,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "steps": result.step_count,
            "total_reward": result.total_reward,
            "final_distance": abs(float(result.observations[-1, 3])),
            "final_position": float(result.observations[-1, 0]),
            "final_velocity": float(result.observations[-1, 1]),
            "mean_action": float(result.actions.mean()),
        }
        for seed, result in zip(seeds, results)
    ]
    return asdict(summarize_episodes(results)), records


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config: dict[str, Any] = json.load(file)
    if int(config["training_updates"]) <= 0 or int(config["episodes_per_update"]) <= 0:
        raise ValueError("training_updates and episodes_per_update must be positive")
    if int(config["evaluation_episode_count"]) <= 0:
        raise ValueError("evaluation_episode_count must be positive")
    training_start = int(config["seed"])
    training_end = training_start + int(config["training_updates"]) * int(config["episodes_per_update"])
    evaluation_start = int(config["evaluation_seed"])
    evaluation_seeds = range(
        evaluation_start, evaluation_start + int(config["evaluation_episode_count"])
    )
    if set(range(training_start, training_end)).intersection(evaluation_seeds):
        raise ValueError("training and evaluation seed ranges must not overlap")

    # Tiny networks and small batches are faster with one CPU thread.
    torch.set_num_threads(1)
    torch.manual_seed(training_start)
    environment = make_environment(config)
    if not np.array_equal(environment.action_space.low, [-1.0]) or not np.array_equal(
        environment.action_space.high, [1.0]
    ):
        raise ValueError("the discrete action set requires environment bounds [-1, 1]")
    model = DiscreteActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    summaries: dict[str, dict[str, float | int]] = {}
    evaluation_records: dict[str, list[dict[str, float | int | bool]]] = {}
    summaries["initial_greedy"], evaluation_records["initial_greedy"] = evaluate(
        environment, GreedyDiscretePolicy(model), evaluation_seeds
    )
    training_history: list[dict[str, float | int]] = []
    training_episode_records: list[dict[str, float | int | bool]] = []
    for update_index in range(int(config["training_updates"])):
        episode_start = training_start + update_index * int(config["episodes_per_update"])
        batch = collect_episodes(
            environment,
            model,
            episode_seeds=range(episode_start, episode_start + int(config["episodes_per_update"])),
            gamma=float(config["gamma"]),
            gae_lambda=float(config["gae_lambda"]),
            reward_scale=float(config["reward_scale"]),
        )
        training_episode_records.extend(
            {"update": update_index + 1, **record}
            for record in batch.episode_records
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
        training_history.append(
            {
                "update": update_index + 1,
                "steps_collected": int(batch.actions.numel()),
                "training_success_rate": sum(
                    bool(record["success"]) for record in batch.episode_records
                ) / len(batch.episode_records),
                "training_mean_reward": float(np.mean([
                    float(record["total_reward"]) for record in batch.episode_records
                ])),
                **metrics,
            }
        )
        if (update_index + 1) % 5 == 0:
            print(
                f"update={update_index + 1}, "
                f"training_success={training_history[-1]['training_success_rate']:.2f}, "
                f"mean_reward={training_history[-1]['training_mean_reward']:.2f}"
            )

    summaries["trained_greedy"], evaluation_records["trained_greedy"] = evaluate(
        environment, GreedyDiscretePolicy(model), evaluation_seeds
    )
    summaries["random"], evaluation_records["random"] = evaluate(
        environment,
        RandomPolicy(environment.action_space, seed=evaluation_start + 10000),
        evaluation_seeds,
    )
    summaries["proportional"], evaluation_records["proportional"] = evaluate(
        environment,
        ProportionalReachPolicy(float(config["proportional_gain"]), environment.action_space),
        evaluation_seeds,
    )
    environment.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "trained_model.pt")
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "experiment_name": config["experiment_name"],
                "training_seed_range": [training_start, training_end - 1],
                "evaluation_seed_range": [evaluation_seeds.start, evaluation_seeds.stop - 1],
                "action_values": ACTION_VALUES,
                "evaluation_mode": "greedy actor for learned policies",
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
        writer = csv.DictWriter(file, fieldnames=["policy", *evaluation_records["random"][0]])
        writer.writeheader()
        for name, records in evaluation_records.items():
            for record in records:
                writer.writerow({"policy": name, **record})
    with (OUTPUT_DIR / "training_episodes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(training_episode_records[0]))
        writer.writeheader()
        writer.writerows(training_episode_records)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
