import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robot_learning.config import PointRobotConfig
from robot_learning.point_robot import (
    analyze_trajectory,
    simulate_constant_velocity,
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "001_point_robot.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "001_point_robot"


def load_config() -> PointRobotConfig:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return PointRobotConfig.from_mapping(json.load(file))


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

    random.seed(config.seed)
    np.random.seed(config.seed)

    trajectory = simulate_constant_velocity(
        initial_position=np.array(config.initial_position),
        velocity=np.array(config.velocity),
        dt=config.dt,
        steps=config.steps,
    )
    stats = analyze_trajectory(trajectory, dt=config.dt)
    final_position = trajectory[-1]
    results = {
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "trajectory_shape": list(trajectory.shape),
        "final_position": trajectory[-1].tolist(),
        "displacement": stats.displacement.tolist(),
        "displacement_distance": stats.displacement_distance,
        "path_length": stats.path_length,
        "average_speed": stats.average_speed,
    }

    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    step_velocities = np.diff(trajectory, axis=0) / config.dt
    step_speeds = np.linalg.norm(step_velocities, axis=1)
    times = np.arange(1, trajectory.shape[0]) * config.dt

    fig, (trajectory_ax, speed_ax) = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )

    trajectory_ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        label="trajectory",
    )
    trajectory_ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        color="green",
        label="start",
        zorder=3,
    )
    trajectory_ax.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        color="red",
        label="end",
        zorder=3,
    )
    trajectory_ax.set(
        title="2D Point Robot Trajectory",
        xlabel="x position",
        ylabel="y position",
    )
    trajectory_ax.axis("equal")
    trajectory_ax.grid(alpha=0.3)
    trajectory_ax.legend()

    speed_ax.plot(times, step_speeds, label="step speed")
    speed_ax.axhline(
        stats.average_speed,
        color="orange",
        linestyle="--",
        label="average speed",
    )
    speed_ax.set(
        title="Speed over Time",
        xlabel="time",
        ylabel="speed",
    )
    speed_ax.grid(alpha=0.3)
    speed_ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "trajectory.png",
        dpi=150,
    )
    plt.close(fig)

    logger.info("配置文件：%s", CONFIG_PATH)
    logger.info("随机种子：%s", config.seed)
    logger.info("轨迹形状：%s", trajectory.shape)
    logger.info("最终位置：%s", final_position)
    logger.info("结果文件：%s", OUTPUT_DIR / "results.json")
    logger.info("位移距离：%.4f", stats.displacement_distance)
    logger.info("总路程：%.4f", stats.path_length)
    logger.info("平均速率：%.4f", stats.average_speed)


if __name__ == "__main__":
    main()
