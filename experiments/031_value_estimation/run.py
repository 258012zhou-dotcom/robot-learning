"""Compare exact V/Q/Advantage with Monte Carlo estimates."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_learning.value_estimation import (
    analyze_policy_values,
    estimate_discrete_action_values,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "031_value_estimation.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "031_value_estimation"


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    """Calculate known values, then estimate them from sampled Episodes."""
    config = load_json(CONFIG_PATH)
    gamma = float(config["discount_factor"])
    safe_probability = float(config["policy"]["safe_probability"])
    risky_probability = float(config["policy"]["risky_probability"])
    policy_probabilities = np.asarray(
        [safe_probability, risky_probability], dtype=np.float64
    )
    safe_q = float(config["safe"]["immediate_reward"])
    risky_config = config["risky"]
    risky_terminal_expectation = (
        float(risky_config["success_probability"])
        * float(risky_config["success_reward"])
        + (1.0 - float(risky_config["success_probability"]))
        * float(risky_config["failure_reward"])
    )
    risky_q = float(risky_config["immediate_reward"]) + (
        gamma * risky_terminal_expectation
    )
    exact_q = np.asarray([safe_q, risky_q], dtype=np.float64)
    exact_analysis = analyze_policy_values(exact_q, policy_probabilities)

    sample_counts = [int(value) for value in config["sample_counts"]]
    if not sample_counts or any(value <= 0 for value in sample_counts):
        raise ValueError("sample_counts must contain positive integers")
    if sample_counts != sorted(set(sample_counts)):
        raise ValueError("sample_counts must be unique and increasing")

    rng = np.random.default_rng(int(config["seed"]))
    maximum_count = sample_counts[-1]
    action_ids = (rng.random(maximum_count) >= safe_probability).astype(
        np.int64
    )
    risky_successes = (
        rng.random(maximum_count)
        < float(risky_config["success_probability"])
    )
    returns = np.full(maximum_count, safe_q, dtype=np.float64)
    risky_rows = action_ids == 1
    risky_terminal_rewards = np.where(
        risky_successes,
        float(risky_config["success_reward"]),
        float(risky_config["failure_reward"]),
    )
    returns[risky_rows] = float(risky_config["immediate_reward"]) + (
        gamma * risky_terminal_rewards[risky_rows]
    )

    estimates = []
    for sample_count in sample_counts:
        estimated_q, action_counts = estimate_discrete_action_values(
            action_ids[:sample_count],
            returns[:sample_count],
            action_count=2,
        )
        estimated_analysis = analyze_policy_values(
            estimated_q,
            policy_probabilities,
        )
        sampled_state_value = float(np.mean(returns[:sample_count]))
        estimates.append(
            {
                "sample_count": sample_count,
                "action_sample_counts": {
                    "safe": int(action_counts[0]),
                    "risky": int(action_counts[1]),
                },
                "monte_carlo_state_value": sampled_state_value,
                "state_value_from_estimated_q": (
                    estimated_analysis.state_value
                ),
                "estimated_q": {
                    "safe": float(estimated_q[0]),
                    "risky": float(estimated_q[1]),
                },
                "estimated_advantage": {
                    "safe": float(estimated_analysis.advantages[0]),
                    "risky": float(estimated_analysis.advantages[1]),
                },
                "absolute_state_value_error": abs(
                    sampled_state_value - exact_analysis.state_value
                ),
            }
        )

    output = {
        "experiment_name": config["experiment_name"],
        "status": "value_estimation_complete",
        "discount_factor": gamma,
        "action_order": ["safe", "risky"],
        "policy_probabilities": policy_probabilities.tolist(),
        "exact": {
            "q": {"safe": float(exact_q[0]), "risky": float(exact_q[1])},
            "state_value": exact_analysis.state_value,
            "advantage": {
                "safe": float(exact_analysis.advantages[0]),
                "risky": float(exact_analysis.advantages[1]),
            },
            "policy_weighted_advantage": float(
                np.dot(policy_probabilities, exact_analysis.advantages)
            ),
        },
        "monte_carlo_estimates": estimates,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(
        f"exact: V={exact_analysis.state_value:.3f}, "
        f"Q_safe={exact_q[0]:.3f}, Q_risky={exact_q[1]:.3f}, "
        f"A_safe={exact_analysis.advantages[0]:.3f}, "
        f"A_risky={exact_analysis.advantages[1]:.3f}"
    )
    for row in estimates:
        print(
            f"samples={row['sample_count']:5d}: "
            f"V_mc={row['monte_carlo_state_value']:.3f}, "
            f"Q_safe={row['estimated_q']['safe']:.3f}, "
            f"Q_risky={row['estimated_q']['risky']:.3f}, "
            f"abs_V_error={row['absolute_state_value_error']:.3f}"
        )


if __name__ == "__main__":
    main()
