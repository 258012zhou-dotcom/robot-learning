"""Show how an unsupported action value enters a fixed-data TD backup."""

import json
from pathlib import Path

import numpy as np

from robot_learning.q_learning import update_q_table


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/042_offline_q_extrapolation.json"
OUTPUT_DIR = ROOT / "outputs/042_offline_q_extrapolation"


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    gamma = float(config["discount_factor"])
    learning_rate = float(config["learning_rate"])
    observed_reward = float(config["observed_terminal_reward"])
    unsupported_initial_q = float(config["unobserved_initial_q"])
    if learning_rate != 1.0 or unsupported_initial_q <= observed_reward:
        raise ValueError("the demonstration requires a full backup and an optimistic unseen action")

    # This is the entire training dataset. Action 1 at S1 never occurs in it.
    fixed_dataset = [
        {"state": "S0", "action": 0, "reward": 0.0, "next_state": "S1"},
        {"state": "S1", "action": 0, "reward": observed_reward, "next_state": None},
    ]
    q_table = np.zeros((2, 2), dtype=np.float64)
    q_table[1, 1] = unsupported_initial_q
    unseen_before = float(q_table[1, 1])

    # First learn the terminal transition present in the dataset.
    update_q_table(
        q_table, 1, 0, observed_reward, None,
        discount_factor=gamma, learning_rate=learning_rate,
    )
    supported_only_target = gamma * float(q_table[1, 0])

    # Ordinary Q-learning maximizes over both actions, including the unseen one.
    naive_update = update_q_table(
        q_table, 0, 0, 0.0, 1,
        discount_factor=gamma, learning_rate=learning_rate,
    )
    if not np.isclose(q_table[1, 1], unseen_before):
        raise RuntimeError("the fixed dataset unexpectedly updated the unseen action")
    if not np.isclose(naive_update.target, gamma * unsupported_initial_q):
        raise RuntimeError("the unsupported value did not enter the TD target")
    if not supported_only_target < naive_update.target:
        raise RuntimeError("the proposed comparison no longer illustrates extrapolation")

    # These are possible worlds for explanation, never samples used in training.
    possible_worlds = [{
        "unobserved_action_reward": float(reward),
        "true_optimal_s0_return": gamma * max(observed_reward, float(reward)),
    } for reward in config["hypothetical_unobserved_rewards"]]
    report = {
        "experiment_name": config["experiment_name"],
        "protocol": config,
        "fixed_training_dataset": fixed_dataset,
        "s1_action_1_sample_count": 0,
        "s1_action_1_q_before_and_after": [unseen_before, float(q_table[1, 1])],
        "ordinary_max_backup_target_at_s0": naive_update.target,
        "observed_actions_only_reference_target_at_s0": supported_only_target,
        "hypothetical_worlds_not_used_for_training": possible_worlds,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
