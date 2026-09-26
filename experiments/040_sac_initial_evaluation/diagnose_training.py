"""Audit which target directions reach replay batches in experiment 040."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.sac_evaluation import DeterministicSACPolicy
from robot_learning.sac_models import SquashedGaussianActor, TwinQCritic
from robot_learning.sac_online import OnlineReplayBuffer, run_online_steps
from robot_learning.sac_training import make_target_critic


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/040_sac_initial_evaluation.json"
MODEL_PATH = ROOT / "experiments/017_mujoco_step/point_robot.xml"
OUTPUT_DIR = ROOT / "outputs/040_sac_initial_evaluation"


class AuditedReplayBuffer(OnlineReplayBuffer):
    """Count sampled directions without changing the sampled batch or RNG."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.negative_per_update: list[int] = []
        self.positive_per_update: list[int] = []

    def sample(self, batch_size: int, rng: np.random.Generator):
        batch = super().sample(batch_size, rng)
        targets = batch.observations[:, 2]
        self.negative_per_update.append(int(np.count_nonzero(targets < 0)))
        self.positive_per_update.append(int(np.count_nonzero(targets > 0)))
        return batch


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
    seed = int(config["training_seed"])
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    actor = SquashedGaussianActor()
    critic = TwinQCritic()
    target_critic = make_target_critic(critic)
    replay = AuditedReplayBuffer(int(config["replay_capacity"]))
    environment = make_environment(config)
    try:
        # A seeded reset is only a probe; run_online_steps resets again.
        negative_start, _ = environment.reset(seed=seed + 1)
        initial_negative_action = float(DeterministicSACPolicy(actor)(negative_start)[0])
        training = run_online_steps(
            environment, actor, critic, target_critic,
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
        trained_negative_action = float(DeterministicSACPolicy(actor)(negative_start)[0])
    finally:
        environment.close()

    transitions = list(replay.transitions)
    negative = [item for item in transitions if item.observation[2] < 0]
    positive = [item for item in transitions if item.observation[2] > 0]
    if len(transitions) != training.environment_steps:
        raise RuntimeError("replay capacity hid some collected transitions")
    if len(replay.negative_per_update) != training.gradient_updates:
        raise RuntimeError("audited batch count differs from gradient updates")
    if any(pos + neg != int(config["batch_size"]) for pos, neg in zip(
        replay.positive_per_update, replay.negative_per_update, strict=True,
    )):
        raise RuntimeError("a sampled transition has an unknown target direction")

    reference_path = OUTPUT_DIR / "results.json"
    reference_matches = None
    if reference_path.exists():
        with reference_path.open(encoding="utf-8") as file:
            reference = json.load(file)["training"]
        actual = asdict(training)
        reference_matches = all(
            np.isclose(actual[key], value, rtol=1e-6, atol=1e-8)
            for key, value in reference.items()
        )
        if not reference_matches:
            raise RuntimeError("audit did not reproduce the original training summary")

    report = {
        "reference_training_matches": reference_matches,
        "training": asdict(training),
        "collected": {
            "positive_target": len(positive),
            "negative_target": len(negative),
            "episode_end_steps": [
                index + 1 for index, item in enumerate(transitions)
                if item.terminated or item.truncated
            ],
        },
        "sampled": {
            "positive_target_draws": sum(replay.positive_per_update),
            "negative_target_draws": sum(replay.negative_per_update),
            "updates_with_negative": sum(count > 0 for count in replay.negative_per_update),
            "first_update_with_negative": next(
                (index + 1 for index, count in enumerate(replay.negative_per_update) if count > 0),
                None,
            ),
        },
        "negative_target_actions": {
            "positive": int(sum(item.action[0] > 0 for item in negative)),
            "negative": int(sum(item.action[0] < 0 for item in negative)),
            "mean": float(np.mean([item.action[0] for item in negative])) if negative else None,
            "first": float(negative[0].action[0]) if negative else None,
            "last": float(negative[-1].action[0]) if negative else None,
            "final_position": float(negative[-1].next_observation[0]) if negative else None,
        },
        "negative_start_deterministic_action": {
            "initial": initial_negative_action,
            "trained": trained_negative_action,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "training_diagnosis.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
