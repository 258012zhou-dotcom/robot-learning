"""运行二维位置观测的滑动平均过滤实验。"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from robot_learning.position_buffer import PositionBuffer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "002_position_filter.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "002_position_filter"


def load_config() -> dict[str, Any]:
    """读取实验配置。"""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def calculate_rmse(estimates: np.ndarray, true_position: np.ndarray) -> float:
    """计算一组二维位置估计相对真实位置的 RMSE。"""
    squared_errors = np.sum((estimates - true_position) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_errors)))


def main() -> None:
    """生成带噪位置观测，过滤并保存结果。"""
    config = load_config()
    true_position = np.asarray(config["true_position"], dtype=float)
    observation_count = int(config["observation_count"])
    noise_std = float(config["noise_std"])
    buffer = PositionBuffer(capacity=int(config["buffer_capacity"]))
    rng = np.random.default_rng(int(config["seed"]))

    observations = true_position + rng.normal(
        loc=0.0,
        scale=noise_std,
        size=(observation_count, 2),
    )
    filtered_positions = np.empty_like(observations)
    for index, observation in enumerate(observations):
        buffer.append(observation)
        filtered_positions[index] = buffer.mean()

    raw_errors = np.linalg.norm(observations - true_position, axis=1)
    filtered_errors = np.linalg.norm(filtered_positions - true_position, axis=1)
    raw_rmse = calculate_rmse(observations, true_position)
    filtered_rmse = calculate_rmse(filtered_positions, true_position)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "true_position": true_position.tolist(),
        "observation_count": observation_count,
        "noise_std": noise_std,
        "buffer_capacity": config["buffer_capacity"],
        "raw_rmse": raw_rmse,
        "filtered_rmse": filtered_rmse,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    fig, axis = plt.subplots(figsize=(9, 4))
    axis.plot(raw_errors, alpha=0.7, label="raw observation error")
    axis.plot(filtered_errors, label="filtered position error")
    axis.set(
        title="Position Error: Raw Observation vs Sliding Average",
        xlabel="observation index",
        ylabel="Euclidean error",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "filter_result.png", dpi=150)
    plt.close(fig)

    print(f"原始观测 RMSE: {raw_rmse:.4f}")
    print(f"过滤后 RMSE: {filtered_rmse:.4f}")


if __name__ == "__main__":
    main()
