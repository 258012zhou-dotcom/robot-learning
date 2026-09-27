"""Compare fixed datasets with and without coverage of one S1 action."""

import json
from pathlib import Path

import numpy as np

from robot_learning.q_learning import update_q_table


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/043_offline_data_coverage.json"
OUTPUT_DIR = ROOT / "outputs/043_offline_data_coverage"


def evaluate_fixed_dataset(config: dict, extra_reward: float | None) -> dict:
    """Read a dataset assembled before learning; never query an environment."""
    gamma = float(config["discount_factor"])
    initial_q = float(config["unseen_action_initial_q"])
    observed_reward = float(config["observed_terminal_reward"])
    learning_rate = float(config["learning_rate"])
    dataset = [
        {"state": "S0", "action": 0, "reward": 0.0, "next_state": "S1"},
        {"state": "S1", "action": 0, "reward": observed_reward, "next_state": None},
    ]
    if extra_reward is not None:
        dataset.append({
            "state": "S1", "action": 1, "reward": extra_reward, "next_state": None,
        })

    table = np.zeros((2, 2), dtype=np.float64)
    table[1, 1] = initial_q
    # Learn only the terminal transitions actually present in this fixed dataset.
    for row in dataset[1:]:
        update_q_table(
            table, 1, int(row["action"]), float(row["reward"]), None,
            discount_factor=gamma, learning_rate=learning_rate,
        )
    predecessor = update_q_table(
        table, 0, 0, 0.0, 1,
        discount_factor=gamma, learning_rate=learning_rate,
    )
    expected_unseen_q = initial_q if extra_reward is None else extra_reward
    if not np.isclose(table[1, 1], expected_unseen_q):
        raise RuntimeError("the S1 action-1 value disagrees with dataset coverage")
    if not np.isclose(predecessor.target, gamma * max(observed_reward, expected_unseen_q)):
        raise RuntimeError("the predecessor backup is inconsistent")
    return {
        "fixed_dataset": dataset,
        "s1_action_1_sample_count": int(extra_reward is not None),
        "s1_action_1_q_after": float(table[1, 1]),
        "s0_action_0_td_target": predecessor.target,
    }


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    gamma = float(config["discount_factor"])
    learning_rate = float(config["learning_rate"])
    rewards = [float(value) for value in config["covered_unseen_action_rewards"]]
    if (
        not 0.0 <= gamma <= 1.0
        or learning_rate != 1.0
        or len(rewards) != 2
        or not all(np.isfinite(value) for value in rewards)
        or rewards[0] == rewards[1]
    ):
        raise ValueError("the fixed-data comparison requires valid paired settings")

    sparse = evaluate_fixed_dataset(config, None)
    covered = [evaluate_fixed_dataset(config, reward) for reward in rewards]
    if any(item["fixed_dataset"][:2] != sparse["fixed_dataset"] for item in covered):
        raise RuntimeError("the shared part of the fixed dataset changed")
    if any(item["s1_action_1_sample_count"] != 1 for item in covered):
        raise RuntimeError("the coverage conditions are incomplete")
    report = {
        "experiment_name": config["experiment_name"],
        "protocol": config,
        "no_action_1_data": sparse,
        "action_1_data_precollected": covered,
        "training_environment_interactions": 0,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
