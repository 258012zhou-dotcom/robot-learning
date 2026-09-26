"""Early SAC checkpoint: paired held-out evaluation before and after training."""

import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from robot_learning.gymnasium_rollout import ProportionalReachPolicy
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.sac_evaluation import (
    DeterministicSACPolicy,
    evaluate_reach_policy,
)
from robot_learning.sac_models import SquashedGaussianActor, TwinQCritic
from robot_learning.sac_online import OnlineReplayBuffer, run_online_steps
from robot_learning.sac_training import make_target_critic


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/040_sac_initial_evaluation.json"
MODEL_PATH = ROOT / "experiments/017_mujoco_step/point_robot.xml"
OUTPUT_DIR = ROOT / "outputs/040_sac_initial_evaluation"


def make_environment(config: dict) -> PointRobotReachEnv:
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


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    training_seed = int(config["training_seed"])
    training_steps = int(config["training_steps"])
    evaluation_seed = int(config["evaluation_seed"])
    evaluation_count = int(config["evaluation_episode_count"])
    if training_steps <= 0 or evaluation_count <= 0:
        raise ValueError("training steps and evaluation count must be positive")
    evaluation_seeds = range(evaluation_seed, evaluation_seed + evaluation_count)
    # At most one new training Episode can begin per environment step.
    possible_training_seeds = range(training_seed, training_seed + training_steps + 1)
    if set(evaluation_seeds).intersection(possible_training_seeds):
        raise ValueError("training and evaluation seeds overlap")

    torch.set_num_threads(1)
    torch.manual_seed(training_seed)
    actor = SquashedGaussianActor()
    critic = TwinQCritic()
    target_critic = make_target_critic(critic)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(config["learning_rate"]))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(config["learning_rate"]))
    replay = OnlineReplayBuffer(capacity=int(config["replay_capacity"]))
    training_environment = make_environment(config)
    evaluation_environment = make_environment(config)
    try:
        initial_summary, initial_records = evaluate_reach_policy(
            evaluation_environment, DeterministicSACPolicy(actor), evaluation_seeds,
        )
        training = run_online_steps(
            training_environment, actor, critic, target_critic,
            actor_optimizer, critic_optimizer, replay,
            steps=training_steps,
            random_steps=int(config["random_steps"]),
            batch_size=int(config["batch_size"]),
            seed=training_seed,
            rng=np.random.default_rng(training_seed),
            gamma=float(config["gamma"]),
            alpha=float(config["alpha"]),
            tau=float(config["tau"]),
        )
        trained_summary, trained_records = evaluate_reach_policy(
            evaluation_environment, DeterministicSACPolicy(actor), evaluation_seeds,
        )
        proportional_summary, proportional_records = evaluate_reach_policy(
            evaluation_environment,
            ProportionalReachPolicy(float(config["proportional_gain"]), evaluation_environment.action_space),
            evaluation_seeds,
        )
    finally:
        training_environment.close()
        evaluation_environment.close()

    records_by_policy = {
        "initial_sac": initial_records,
        "trained_sac": trained_records,
        "proportional": proportional_records,
    }
    for records in zip(*records_by_policy.values(), strict=True):
        if len({record["seed"] for record in records}) != 1 or len({record["target_position"] for record in records}) != 1:
            raise RuntimeError("policies were not evaluated on matching tasks")

    report = {
        "protocol": config,
        "training": asdict(training),
        "evaluation": {
            "initial_sac": asdict(initial_summary),
            "trained_sac": asdict(trained_summary),
            "proportional": asdict(proportional_summary),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "episodes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["policy", *initial_records[0].keys()])
        writer.writeheader()
        for policy_name, records in records_by_policy.items():
            for record in records:
                writer.writerow({"policy": policy_name, **record})
    print(json.dumps(report["evaluation"], ensure_ascii=False, indent=2))
    print(f"Training: {training.environment_steps} steps, {training.gradient_updates} updates")


if __name__ == "__main__":
    main()
