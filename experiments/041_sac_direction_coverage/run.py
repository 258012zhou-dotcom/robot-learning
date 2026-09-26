"""Controlled small-budget comparison of SAC target-direction coverage."""

import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from robot_learning.gymnasium_rollout import ProportionalReachPolicy
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.sac_direction_schedule import ScheduledTargetEnvironment
from robot_learning.sac_evaluation import DeterministicSACPolicy, evaluate_reach_policy
from robot_learning.sac_models import SquashedGaussianActor, TwinQCritic
from robot_learning.sac_online import OnlineReplayBuffer, run_online_steps
from robot_learning.sac_training import make_target_critic


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/041_sac_direction_coverage.json"
MODEL_PATH = ROOT / "experiments/017_mujoco_step/point_robot.xml"
OUTPUT_DIR = ROOT / "outputs/041_sac_direction_coverage"


class CountingReplayBuffer(OnlineReplayBuffer):
    """Measure target-direction exposure without changing sampled batches."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.negative_draws = 0
        self.positive_draws = 0

    def sample(self, batch_size: int, rng: np.random.Generator):
        batch = super().sample(batch_size, rng)
        targets = batch.observations[:, 2]
        self.negative_draws += int(np.count_nonzero(targets < 0))
        self.positive_draws += int(np.count_nonzero(targets > 0))
        return batch


def make_environment(config: dict, *, max_episode_steps: int) -> PointRobotReachEnv:
    return PointRobotReachEnv(
        MODEL_PATH,
        frame_skip=int(config["frame_skip"]),
        max_episode_steps=max_episode_steps,
        minimum_target_distance=float(config["minimum_target_distance"]),
        maximum_target_distance=float(config["maximum_target_distance"]),
        success_tolerance=float(config["success_tolerance"]),
        velocity_tolerance=float(config["velocity_tolerance"]),
        action_penalty_weight=float(config["action_penalty_weight"]),
    )


def train_arm(
    config: dict, *, alternate_direction: bool, evaluation_seeds: range,
    initial_parameters: tuple[torch.Tensor, ...] | None,
) -> tuple[dict, dict, list[dict], tuple[torch.Tensor, ...]]:
    seed = int(config["training_seed"])
    torch.manual_seed(seed)
    actor = SquashedGaussianActor()
    critic = TwinQCritic()
    parameters = tuple(
        parameter.detach().clone() for model in (actor, critic)
        for parameter in model.parameters()
    )
    if initial_parameters is not None and not all(
        torch.equal(first, second)
        for first, second in zip(initial_parameters, parameters, strict=True)
    ):
        raise RuntimeError("the two training arms did not start from the same weights")

    target_critic = make_target_critic(critic)
    replay = CountingReplayBuffer(int(config["replay_capacity"]))
    magnitudes = tuple(float(value) for value in config["target_magnitudes"])
    training_environment = ScheduledTargetEnvironment(
        make_environment(config, max_episode_steps=int(config["training_episode_steps"])),
        target_magnitudes=magnitudes,
        alternate_direction=alternate_direction,
    )
    evaluation_environment = make_environment(
        config, max_episode_steps=int(config["evaluation_episode_steps"]),
    )
    try:
        training = run_online_steps(
            training_environment, actor, critic, target_critic,
            torch.optim.Adam(actor.parameters(), lr=float(config["learning_rate"])),
            torch.optim.Adam(critic.parameters(), lr=float(config["learning_rate"])),
            replay,
            steps=int(config["training_steps"]),
            random_steps=int(config["random_steps"]),
            batch_size=int(config["batch_size"]),
            seed=seed,
            rng=np.random.default_rng(seed),
            gamma=float(config["gamma"]),
            alpha=float(config["alpha"]),
            tau=float(config["tau"]),
        )
        evaluation, records = evaluate_reach_policy(
            evaluation_environment, DeterministicSACPolicy(actor), evaluation_seeds,
        )
    finally:
        training_environment.close()
        evaluation_environment.close()

    # Early success would shift the episode schedule and break this paired test.
    if len(training_environment.scheduled_targets) != len(magnitudes):
        raise RuntimeError("an arm did not run the planned number of training episodes")
    if training.completed_episodes != len(magnitudes):
        raise RuntimeError("a training episode did not end at the planned boundary")
    np.testing.assert_allclose(
        np.abs(training_environment.scheduled_targets), magnitudes,
    )
    if replay.positive_draws + replay.negative_draws != training.gradient_updates * int(config["batch_size"]):
        raise RuntimeError("replay draws did not match the number of updates")
    counts = {
        "positive_transitions": sum(item.observation[2] > 0 for item in replay.transitions),
        "negative_transitions": sum(item.observation[2] < 0 for item in replay.transitions),
        "positive_replay_draws": replay.positive_draws,
        "negative_replay_draws": replay.negative_draws,
        "scheduled_targets": training_environment.scheduled_targets,
    }
    # NumPy booleans sum to NumPy integers, which JSON cannot encode.
    counts["positive_transitions"] = int(counts["positive_transitions"])
    counts["negative_transitions"] = int(counts["negative_transitions"])
    return {"training": asdict(training), "coverage": counts}, asdict(evaluation), records, parameters


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    magnitudes = config["target_magnitudes"]
    if len(magnitudes) % 2 or int(config["training_steps"]) != len(magnitudes) * int(config["training_episode_steps"]):
        raise ValueError("training steps must exactly cover paired target magnitudes")
    if any(magnitudes[index] != magnitudes[index + 1] for index in range(0, len(magnitudes), 2)):
        raise ValueError("target magnitudes must occur in equal pairs")
    evaluation_seeds = range(
        int(config["evaluation_seed"]),
        int(config["evaluation_seed"]) + int(config["evaluation_episode_count"]),
    )
    if not evaluation_seeds or set(evaluation_seeds).intersection(range(
        int(config["training_seed"]),
        int(config["training_seed"]) + int(config["training_steps"]) + 1,
    )):
        raise ValueError("evaluation seeds must be non-empty and distinct from training")

    torch.set_num_threads(1)
    positive_training, positive_evaluation, positive_records, initial_parameters = train_arm(
        config, alternate_direction=False, evaluation_seeds=evaluation_seeds,
        initial_parameters=None,
    )
    balanced_training, balanced_evaluation, balanced_records, _ = train_arm(
        config, alternate_direction=True, evaluation_seeds=evaluation_seeds,
        initial_parameters=initial_parameters,
    )
    for left, right in zip(positive_records, balanced_records, strict=True):
        if left["seed"] != right["seed"] or left["target_position"] != right["target_position"]:
            raise RuntimeError("the two models were evaluated on different tasks")
    evaluation_environment = make_environment(
        config, max_episode_steps=int(config["evaluation_episode_steps"]),
    )
    try:
        proportional_evaluation, proportional_records = evaluate_reach_policy(
            evaluation_environment,
            ProportionalReachPolicy(float(config["proportional_gain"]), evaluation_environment.action_space),
            evaluation_seeds,
        )
    finally:
        evaluation_environment.close()

    report = {
        "protocol": config,
        "positive_only": {**positive_training, "evaluation": positive_evaluation},
        "balanced": {**balanced_training, "evaluation": balanced_evaluation},
        "proportional": {"evaluation": asdict(proportional_evaluation)},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / "episodes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["policy", *positive_records[0].keys()])
        writer.writeheader()
        for name, records in (
            ("positive_only", positive_records),
            ("balanced", balanced_records),
            ("proportional", proportional_records),
        ):
            for record in records:
                writer.writerow({"policy": name, **record})
    print(json.dumps({
        name: {"coverage": report[name].get("coverage"), "evaluation": report[name]["evaluation"]}
        for name in ("positive_only", "balanced", "proportional")
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
