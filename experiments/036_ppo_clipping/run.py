"""Compare fixed-batch policy updates with and without PPO clipping."""

import json
from pathlib import Path
from typing import Any

import torch

from robot_learning.ppo_clipping import importance_weighted_loss, ppo_clipped_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "036_ppo_clipping.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "036_ppo_clipping"


def train_condition(config: dict[str, Any], *, clipped: bool) -> dict[str, Any]:
    """Reuse the same enumerated old-policy batch for several updates."""
    reward_low = float(config["reward_low"])
    reward_high = float(config["reward_high"])
    old_value = (reward_low + reward_high) / 2.0
    actions = torch.tensor([0.0, 1.0])
    advantages = torch.tensor([reward_low - old_value, reward_high - old_value])
    old_log_probabilities = torch.distributions.Bernoulli(
        logits=torch.tensor(0.0)
    ).log_prob(actions)

    new_logit = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([new_logit], lr=float(config["learning_rate"]))
    steps = int(config["training_steps"])
    record_steps = [int(step) for step in config["record_steps"]]
    if (
        not record_steps
        or record_steps != sorted(set(record_steps))
        or record_steps[0] != 0
        or record_steps[-1] != steps
    ):
        raise ValueError("record_steps must be unique and cover zero and training_steps")

    def snapshot(completed_steps: int) -> dict[str, float | int]:
        probability_high = float(torch.sigmoid(new_logit.detach()))
        return {
            "completed_steps": completed_steps,
            "probability_high": probability_high,
            "ratio_high": probability_high / 0.5,
            "exact_expected_return": (
                reward_low + probability_high * (reward_high - reward_low)
            ),
        }

    checkpoints = [snapshot(0)]
    for step in range(1, steps + 1):
        new_log_probabilities = torch.distributions.Bernoulli(
            logits=new_logit
        ).log_prob(actions)
        if clipped:
            loss = ppo_clipped_loss(
                new_log_probabilities,
                old_log_probabilities,
                advantages,
                float(config["clip_epsilon"]),
            )
        else:
            loss = importance_weighted_loss(
                new_log_probabilities, old_log_probabilities, advantages
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in record_steps:
            checkpoints.append(snapshot(step))

    return {
        "condition": "ppo_clipped" if clipped else "unclipped",
        "checkpoints": checkpoints,
    }


def main() -> None:
    """Save a deterministic mechanism comparison, not a full PPO run."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config: dict[str, Any] = json.load(file)
    if float(config["reward_high"]) <= float(config["reward_low"]):
        raise ValueError("reward_high must exceed reward_low")
    if int(config["training_steps"]) <= 0 or float(config["learning_rate"]) <= 0:
        raise ValueError("training_steps and learning_rate must be positive")
    if not 0.0 < float(config["clip_epsilon"]) < 1.0:
        raise ValueError("clip_epsilon must be between zero and one")

    conditions = [
        train_condition(config, clipped=False),
        train_condition(config, clipped=True),
    ]
    output = {
        "experiment_name": config["experiment_name"],
        "task": "one-state two-action bandit",
        "data": "both actions enumerated once under frozen P(high)=0.5; not a rollout",
        "old_policy_value": (float(config["reward_low"]) + float(config["reward_high"])) / 2.0,
        "conditions": conditions,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    for condition in conditions:
        print(condition["condition"])
        for row in condition["checkpoints"]:
            print(
                f"  updates={row['completed_steps']:2d}, "
                f"P(high)={row['probability_high']:.4f}, "
                f"ratio_high={row['ratio_high']:.4f}"
            )


if __name__ == "__main__":
    main()
