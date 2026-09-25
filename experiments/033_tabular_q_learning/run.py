"""Learn a two-state discrete task with and without exploration."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_learning.q_learning import epsilon_greedy_action, update_q_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "033_tabular_q_learning.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "033_tabular_q_learning"
ACTION_NAMES = (("take_one", "continue"), ("take_zero", "take_two"))


def transition(state: int, action: int) -> tuple[int | None, float]:
    """Return next state and reward for the tiny deterministic MDP."""
    if state == 0:
        return (None, 1.0) if action == 0 else (1, 0.0)
    if state == 1:
        return (None, 0.0) if action == 0 else (None, 2.0)
    raise ValueError("unknown state")


def run_condition(config: dict[str, Any], epsilon: float) -> dict[str, Any]:
    """Train one table and evaluate its final greedy policy separately."""
    rng = np.random.default_rng(int(config["seed"]))
    table = np.zeros((2, 2), dtype=np.float64)
    visits = np.zeros((2, 2), dtype=np.int64)
    for _ in range(int(config["episode_count"])):
        state: int | None = 0
        while state is not None:
            action = epsilon_greedy_action(table[state], epsilon, rng)
            visits[state, action] += 1
            next_state, reward = transition(state, action)
            update_q_table(
                table,
                state,
                action,
                reward,
                next_state,
                discount_factor=float(config["discount_factor"]),
                learning_rate=float(config["learning_rate"]),
            )
            state = next_state

    greedy_actions = np.argmax(table, axis=1)
    rewards = []
    for _ in range(int(config["evaluation_episodes"])):
        state = 0
        total = 0.0
        discount = 1.0
        while state is not None:
            action = int(greedy_actions[state])
            next_state, reward = transition(state, action)
            total += discount * reward
            discount *= float(config["discount_factor"])
            state = next_state
        rewards.append(total)
    return {
        "epsilon": epsilon,
        "q_table": table.tolist(),
        "training_visits": visits.tolist(),
        "greedy_action_names": [
            ACTION_NAMES[state][int(action)]
            for state, action in enumerate(greedy_actions)
        ],
        "greedy_evaluation_mean_return": float(np.mean(rewards)),
    }


def main() -> None:
    """Save paired, reproducible exploration conditions."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    if int(config["episode_count"]) <= 0 or int(config["evaluation_episodes"]) <= 0:
        raise ValueError("episode counts must be positive")
    conditions = [
        run_condition(config, float(epsilon))
        for epsilon in config["exploration_rates"]
    ]
    output = {
        "experiment_name": config["experiment_name"],
        "true_optimal_initial_q": [1.0, 2.0 * float(config["discount_factor"])],
        "conditions": conditions,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    for condition in conditions:
        print(
            f"epsilon={condition['epsilon']:.1f}: "
            f"visits={condition['training_visits']}, "
            f"greedy={condition['greedy_action_names']}, "
            f"mean_return={condition['greedy_evaluation_mean_return']:.3f}"
        )


if __name__ == "__main__":
    main()
