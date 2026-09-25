"""Train a two-action policy and inspect its score-function gradient."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_learning.policy_gradient import reinforce_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "034_policy_gradient.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "034_policy_gradient"


def exact_gradient_statistics(
    probability_high: float,
    reward_low: float,
    reward_high: float,
    baseline: float,
) -> dict[str, float]:
    """Enumerate both actions to check mean and variance of one sample."""
    probabilities = np.asarray([1.0 - probability_high, probability_high])
    rewards = np.asarray([reward_low, reward_high])
    score_gradients = np.asarray([-probability_high, 1.0 - probability_high])
    samples = (rewards - baseline) * score_gradients
    mean = float(np.dot(probabilities, samples))
    variance = float(np.dot(probabilities, (samples - mean) ** 2))
    return {"mean_gradient": mean, "single_sample_variance": variance}


def main() -> None:
    """Compare exact gradient noise, then train from sampled actions."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config: dict[str, Any] = json.load(file)
    reward_low = float(config["reward_low"])
    reward_high = float(config["reward_high"])
    steps = int(config["training_steps"])
    record_steps = [int(step) for step in config["record_steps"]]
    if reward_low >= reward_high or steps <= 0:
        raise ValueError("reward_high must exceed reward_low and steps must be positive")
    if record_steps != sorted(set(record_steps)) or record_steps[0] != 0 or record_steps[-1] != steps:
        raise ValueError("record_steps must be unique and cover zero and training_steps")
    torch.manual_seed(int(config["seed"]))
    logit = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([logit], lr=float(config["learning_rate"]))

    initial_expected_return = (reward_low + reward_high) / 2.0
    gradient_comparison = {
        "without_baseline": exact_gradient_statistics(
            0.5, reward_low, reward_high, 0.0
        ),
        "with_exact_value_baseline": exact_gradient_statistics(
            0.5, reward_low, reward_high, initial_expected_return
        ),
    }

    def snapshot(completed_steps: int) -> dict[str, float | int]:
        probability_high = float(torch.sigmoid(logit.detach()))
        return {
            "completed_steps": completed_steps,
            "probability_high": probability_high,
            "exact_expected_return": (
                reward_low + probability_high * (reward_high - reward_low)
            ),
        }

    checkpoints = [snapshot(0)]
    sample_counts = {"low": 0, "high": 0}
    for step in range(1, steps + 1):
        distribution = torch.distributions.Bernoulli(logits=logit)
        action = distribution.sample()
        is_high = bool(action.item())
        reward = reward_high if is_high else reward_low
        sample_counts["high" if is_high else "low"] += 1
        with torch.no_grad():
            # The toy task's exact V is known. A learned critic comes later.
            baseline = reward_low + torch.sigmoid(logit) * (
                reward_high - reward_low
            )
        advantage = torch.as_tensor(reward) - baseline
        loss = reinforce_loss(distribution.log_prob(action), advantage)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in record_steps:
            checkpoints.append(snapshot(step))

    output = {
        "experiment_name": config["experiment_name"],
        "task": "one state; low action gives 1, high action gives 2",
        "gradient_at_initial_policy": gradient_comparison,
        "training_sample_counts": sample_counts,
        "checkpoints": checkpoints,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    for row in checkpoints:
        print(
            f"steps={row['completed_steps']:3d}: "
            f"P(high)={row['probability_high']:.4f}, "
            f"exact expected return={row['exact_expected_return']:.4f}"
        )
    print("gradient statistics at P(high)=0.5:", gradient_comparison)


if __name__ == "__main__":
    main()
