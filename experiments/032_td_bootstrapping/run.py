"""Compare Monte Carlo and TD(0) value propagation on a tiny chain."""

import json
from pathlib import Path
from typing import Any

from robot_learning.temporal_difference import update_state_value_td


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "032_td_bootstrapping.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "032_td_bootstrapping"


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def absolute_value_error(
    values: dict[str, float],
    true_values: dict[str, float],
) -> float:
    """Return the sum of absolute state-value errors."""
    return sum(abs(values[state] - true_values[state]) for state in values)


def record_values(
    episode: int,
    td_values: dict[str, float],
    monte_carlo_values: dict[str, float],
    true_values: dict[str, float],
) -> dict[str, Any]:
    """Keep one transparent learning checkpoint."""
    return {
        "completed_episodes": episode,
        "td_values": dict(td_values),
        "monte_carlo_values": dict(monte_carlo_values),
        "td_total_absolute_error": absolute_value_error(td_values, true_values),
        "monte_carlo_total_absolute_error": absolute_value_error(
            monte_carlo_values, true_values
        ),
    }


def main() -> None:
    """Run repeated deterministic Episodes with identical learning rates."""
    config = load_json(CONFIG_PATH)
    gamma = float(config["discount_factor"])
    learning_rate = float(config["learning_rate"])
    episode_count = int(config["episode_count"])
    record_episodes = [int(value) for value in config["record_episodes"]]
    if record_episodes != sorted(set(record_episodes)):
        raise ValueError("record_episodes must be unique and increasing")
    if not record_episodes or record_episodes[0] != 0:
        raise ValueError("record_episodes must begin with zero")
    if record_episodes[-1] != episode_count:
        raise ValueError("record_episodes must end at episode_count")

    s0_reward = float(config["chain"]["s0_reward"])
    terminal_reward = float(config["chain"]["s1_terminal_reward"])
    true_values = {
        "s0": s0_reward + gamma * terminal_reward,
        "s1": terminal_reward,
    }
    td_values = {"s0": 0.0, "s1": 0.0}
    monte_carlo_values = {"s0": 0.0, "s1": 0.0}
    checkpoints = [
        record_values(
            0,
            td_values,
            monte_carlo_values,
            true_values,
        )
    ]

    for episode in range(1, episode_count + 1):
        # TD updates online. S0 can only bootstrap from the value S1 had before
        # the current Episode's terminal transition was processed.
        s0_update = update_state_value_td(
            td_values["s0"],
            reward=s0_reward,
            next_value=td_values["s1"],
            discount_factor=gamma,
            learning_rate=learning_rate,
            terminated=False,
        )
        td_values["s0"] = s0_update.updated_value
        s1_update = update_state_value_td(
            td_values["s1"],
            reward=terminal_reward,
            next_value=0.0,
            discount_factor=gamma,
            learning_rate=learning_rate,
            terminated=True,
        )
        td_values["s1"] = s1_update.updated_value

        # Monte Carlo waits for termination, then both states receive their
        # complete, exact reward-to-go from this deterministic Episode.
        for state in ("s0", "s1"):
            error = true_values[state] - monte_carlo_values[state]
            monte_carlo_values[state] += learning_rate * error

        if episode in record_episodes:
            checkpoints.append(
                record_values(
                    episode,
                    td_values,
                    monte_carlo_values,
                    true_values,
                )
            )

    output = {
        "experiment_name": config["experiment_name"],
        "status": "td_bootstrapping_demonstration_complete",
        "chain": "s0 --reward 0--> s1 --reward 1--> terminal",
        "discount_factor": gamma,
        "learning_rate": learning_rate,
        "true_values": true_values,
        "checkpoints": checkpoints,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(f"true values: V(s0)={true_values['s0']:.3f}, V(s1)=1.000")
    for row in checkpoints:
        print(
            f"episodes={row['completed_episodes']:2d}: "
            f"TD=({row['td_values']['s0']:.3f}, "
            f"{row['td_values']['s1']:.3f}), "
            f"MC=({row['monte_carlo_values']['s0']:.3f}, "
            f"{row['monte_carlo_values']['s1']:.3f})"
        )


if __name__ == "__main__":
    main()
