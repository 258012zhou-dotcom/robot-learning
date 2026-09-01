"""Run the first MuJoCo point-robot simulation and save state evidence."""

import csv
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from robot_learning.mujoco_basics import (
    MujocoSimulationResult,
    load_mujoco_model,
    simulate_constant_control,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "017_mujoco_step.json"
MODEL_PATH = EXPERIMENT_DIR / "point_robot.xml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "017_mujoco_step"


def load_config() -> dict[str, Any]:
    """Read simulation settings from JSON."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def save_trajectory_csv(result: MujocoSimulationResult) -> None:
    """Save one row per simulation sample for later analysis."""
    with (OUTPUT_DIR / "trajectory.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(("time", "position", "velocity", "control"))
        writer.writerows(
            zip(
                result.times,
                result.positions,
                result.velocities,
                result.controls,
            )
        )


def save_state_plot(result: MujocoSimulationResult) -> None:
    """Plot control input and the resulting position and velocity."""
    figure, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(result.times, result.controls, color="tab:red")
    axes[0].set_ylabel("control")
    axes[1].plot(result.times, result.positions, color="tab:blue")
    axes[1].set_ylabel("position [m]")
    axes[2].plot(result.times, result.velocities, color="tab:green")
    axes[2].set(xlabel="simulation time [s]", ylabel="velocity [m/s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("MuJoCo point robot under constant motor control")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "state_trajectory.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Compile the model, run the simulation, and save reproducible outputs."""
    config = load_config()
    model = load_mujoco_model(MODEL_PATH)
    result = simulate_constant_control(
        model,
        joint_name=str(config["joint_name"]),
        actuator_name=str(config["actuator_name"]),
        control=float(config["control"]),
        step_count=int(config["step_count"]),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=(
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "run.log", encoding="utf-8"),
        ),
    )
    save_trajectory_csv(result)
    save_state_plot(result)

    summary = {
        "experiment_name": config["experiment_name"],
        "model_dimensions": {
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
        },
        "timestep": model.opt.timestep,
        "step_count": int(config["step_count"]),
        "control": float(config["control"]),
        "final_time": float(result.times[-1]),
        "final_position": float(result.positions[-1]),
        "final_velocity": float(result.velocities[-1]),
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    logging.info("模型维度：nq=%s nv=%s nu=%s", model.nq, model.nv, model.nu)
    logging.info("仿真时间：%.2f s", summary["final_time"])
    logging.info("最终位置：%.4f m", summary["final_position"])
    logging.info("最终速度：%.4f m/s", summary["final_velocity"])
    logging.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
