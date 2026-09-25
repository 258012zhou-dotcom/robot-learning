"""Compare an exact value baseline with a Critic learned from rewards."""

import json
from pathlib import Path
from typing import Any

import torch

from robot_learning.actor_critic import value_prediction_loss
from robot_learning.policy_gradient import reinforce_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "035_actor_critic.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "035_actor_critic"


def train_condition(
    config: dict[str, Any],
    *,
    learned_critic: bool,
) -> dict[str, Any]:
    """Train one policy with an exact or learned state-value baseline."""
    torch.manual_seed(int(config["seed"]))
    reward_low = float(config["reward_low"])
    reward_high = float(config["reward_high"])
    actor_logit = torch.nn.Parameter(torch.tensor(0.0))
    actor_optimizer = torch.optim.SGD(
        [actor_logit], lr=float(config["actor_learning_rate"])
    )
    critic_value = torch.nn.Parameter(torch.tensor(0.0)) if learned_critic else None
    critic_optimizer = (
        torch.optim.SGD(
            [critic_value], lr=float(config["critic_learning_rate"])
        )
        if critic_value is not None
        else None
    )

    def snapshot(completed_steps: int) -> dict[str, float | int | None]:
        probability_high = float(torch.sigmoid(actor_logit.detach()))
        exact_value = reward_low + probability_high * (reward_high - reward_low)
        prediction = (
            float(critic_value.detach()) if critic_value is not None else None
        )
        return {
            "completed_steps": completed_steps,
            "probability_high": probability_high,
            "exact_expected_return": exact_value,
            "critic_prediction": prediction,
            "critic_absolute_error": (
                abs(prediction - exact_value) if prediction is not None else None
            ),
        }

    record_steps = [int(step) for step in config["record_steps"]]
    steps = int(config["training_steps"])
    if record_steps != sorted(set(record_steps)) or record_steps[0] != 0 or record_steps[-1] != steps:
        raise ValueError("record_steps must be unique and cover zero and training_steps")
    checkpoints = [snapshot(0)]
    action_counts = {"low": 0, "high": 0}
    for step in range(1, steps + 1):
        distribution = torch.distributions.Bernoulli(logits=actor_logit)
        action = distribution.sample()
        is_high = bool(action.item())
        action_counts["high" if is_high else "low"] += 1
        reward = torch.tensor(reward_high if is_high else reward_low)

        # The actor evaluates the action using the baseline available before
        # either network is updated with the current reward.
        with torch.no_grad():
            baseline = (
                critic_value.detach().clone()
                if critic_value is not None
                else reward_low + torch.sigmoid(actor_logit) * (
                    reward_high - reward_low
                )
            )
        actor_loss = reinforce_loss(
            distribution.log_prob(action), reward - baseline
        )
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        if critic_value is not None and critic_optimizer is not None:
            # This one-step episode terminates immediately: the reward is the
            # full return, so the TD target has no next-state bootstrap term.
            critic_loss = value_prediction_loss(critic_value, reward)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
        if step in record_steps:
            checkpoints.append(snapshot(step))

    return {
        "baseline_type": "learned_critic" if learned_critic else "exact_value",
        "training_action_counts": action_counts,
        "checkpoints": checkpoints,
    }


def main() -> None:
    """Run both baselines with equal training budgets and save diagnostics."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config: dict[str, Any] = json.load(file)
    if float(config["reward_high"]) <= float(config["reward_low"]):
        raise ValueError("reward_high must exceed reward_low")
    if int(config["training_steps"]) <= 0:
        raise ValueError("training_steps must be positive")
    conditions = [
        train_condition(config, learned_critic=False),
        train_condition(config, learned_critic=True),
    ]
    output = {
        "experiment_name": config["experiment_name"],
        "task": "one-state terminal bandit",
        "critic_target": "sampled terminal reward",
        "conditions": conditions,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    for condition in conditions:
        print(condition["baseline_type"])
        for row in condition["checkpoints"]:
            critic = row["critic_prediction"]
            critic_text = "n/a" if critic is None else f"{critic:.4f}"
            print(
                f"  step={row['completed_steps']:3d}, "
                f"P(high)={row['probability_high']:.4f}, "
                f"exact_V={row['exact_expected_return']:.4f}, "
                f"critic_V={critic_text}"
            )


if __name__ == "__main__":
    main()
