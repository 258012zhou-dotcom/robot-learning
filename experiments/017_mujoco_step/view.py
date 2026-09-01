"""Display the point robot while the program advances MuJoCo in real time."""

import json
from pathlib import Path
import time
from typing import Any

import mujoco
import mujoco.viewer

from robot_learning.mujoco_basics import load_mujoco_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "017_mujoco_step.json"
MODEL_PATH = EXPERIMENT_DIR / "point_robot.xml"


def load_config() -> dict[str, Any]:
    """Read Viewer control and duration settings."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """Advance physics, synchronize the Viewer, and pace wall-clock time."""
    config = load_config()
    model = load_mujoco_model(MODEL_PATH)
    data = mujoco.MjData(model)
    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        str(config["actuator_name"]),
    )
    if actuator_id == -1:
        raise ValueError("configured actuator does not exist")

    control = float(config["viewer_control"])
    duration = float(config["viewer_duration_seconds"])
    data.ctrl[actuator_id] = control

    # launch_passive displays data but leaves stepping and control to this loop.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < duration:
            wall_step_start = time.perf_counter()

            # Physics advances by model.opt.timestep, independent of rendering.
            mujoco.mj_step(model, data)
            viewer.sync()

            # Sleeping affects playback speed only; it does not change physics.
            wall_step_duration = time.perf_counter() - wall_step_start
            remaining_time = model.opt.timestep - wall_step_duration
            if remaining_time > 0.0:
                time.sleep(remaining_time)

    print(f"viewer finished at simulation time {data.time:.2f} s")
    print(f"final position: {float(data.qpos[0]):.4f} m")


if __name__ == "__main__":
    main()
