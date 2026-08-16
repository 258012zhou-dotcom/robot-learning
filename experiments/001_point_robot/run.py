import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robot_learning.point_robot import (
    analyze_trajectory,
    simulate_constant_velocity,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "001_point_robot.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "001_point_robot"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def configure_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "run.log", encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    logger = configure_logging()

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    trajectory = simulate_constant_velocity(
        initial_position=np.array(config["initial_position"]),
        velocity=np.array(config["velocity"]),
        dt=config["dt"],
        steps=config["steps"],
    )
    stats = analyze_trajectory(trajectory, dt=config["dt"])
    final_position = trajectory[-1]
    results = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "trajectory_shape": list(trajectory.shape),
        "final_position": trajectory[-1].tolist(),
        "displacement": stats.displacement.tolist(),
        "displacement_distance": stats.displacement_distance,
        "path_length": stats.path_length,
        "average_speed": stats.average_speed,
    }

    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    plt.plot(trajectory[:, 0], trajectory[:, 1], marker="o", markevery=10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Point Robot Trajectory")
    plt.grid()
    plt.axis("equal")
    plt.savefig(OUTPUT_DIR / "trajectory.png")
    plt.close()

    logger.info("配置文件：%s", CONFIG_PATH)
    logger.info("随机种子：%s", seed)
    logger.info("轨迹形状：%s", trajectory.shape)
    logger.info("最终位置：%s", final_position)
    logger.info("结果文件：%s", OUTPUT_DIR / "results.json")
    logger.info("位移距离：%.4f", stats.displacement_distance)
    logger.info("总路程：%.4f", stats.path_length)
    logger.info("平均速率：%.4f", stats.average_speed)


if __name__ == "__main__":
    main()