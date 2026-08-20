"""运行一维点质量 PID 闭环控制实验。"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from robot_learning.pid_control import PIDGains, simulate_pid_point_mass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "005_pid_control.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "005_pid_control"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def find_settling_time(
    times: np.ndarray,
    errors: np.ndarray,
    tolerance: float,
) -> float | None:
    """返回误差最后一次离开容差带之后的时间。"""
    outside_indices = np.flatnonzero(np.abs(errors) > tolerance)
    if outside_indices.size == 0:
        return float(times[0])
    final_outside = int(outside_indices[-1])
    if final_outside == len(times) - 1:
        return None
    return float(times[final_outside + 1])


def main() -> None:
    config = load_config()
    dt = float(config["dt"])
    steps = round(float(config["duration"]) / dt)
    target = float(config["target_position"])
    tolerance = float(config["settling_tolerance"])
    gains = PIDGains(
        kp=float(config["kp"]),
        ki=float(config["ki"]),
        kd=float(config["kd"]),
    )

    simulation = simulate_pid_point_mass(
        initial_position=float(config["initial_position"]),
        initial_velocity=float(config["initial_velocity"]),
        target_position=target,
        mass=float(config["mass"]),
        dt=dt,
        steps=steps,
        gains=gains,
        output_limit=float(config["output_limit"]),
        integral_limit=float(config["integral_limit"]),
    )
    errors = target - simulation.positions
    settling_time = find_settling_time(
        simulation.times,
        errors,
        tolerance,
    )
    overshoot = max(float(np.max(simulation.positions) - target), 0.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment_name": config["experiment_name"],
        "final_position": float(simulation.positions[-1]),
        "final_velocity": float(simulation.velocities[-1]),
        "final_absolute_error": abs(float(errors[-1])),
        "maximum_overshoot": overshoot,
        "settling_tolerance": tolerance,
        "settling_time": settling_time,
        "maximum_absolute_control": float(
            np.max(np.abs(simulation.controls))
        ),
    }
    with (OUTPUT_DIR / "results.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    figure, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(simulation.times, simulation.positions, label="position")
    axes[0].axhline(target, color="black", linestyle="--", label="target")
    axes[0].set_ylabel("position")
    axes[0].legend()
    axes[1].plot(simulation.times, simulation.velocities)
    axes[1].set_ylabel("velocity")
    axes[2].plot(simulation.times[:-1], simulation.controls)
    axes[2].set_ylabel("control")
    axes[2].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "pid_response.png", dpi=150)
    plt.close(figure)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("最终位置：%.4f", results["final_position"])
    logger.info("最终绝对误差：%.4f", results["final_absolute_error"])
    logger.info("最大超调：%.4f", results["maximum_overshoot"])
    logger.info("调节时间：%s", results["settling_time"])
    logger.info(
        "最大绝对控制量：%.4f",
        results["maximum_absolute_control"],
    )


if __name__ == "__main__":
    main()
