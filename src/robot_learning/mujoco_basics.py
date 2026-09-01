"""Minimal MuJoCo state access and deterministic simulation utilities.

This module keeps the first MuJoCo lesson focused on one loop:

``write data.ctrl -> mj_step(model, data) -> read data.qpos/qvel``

Rendering and Gymnasium wrappers are intentionally left for later lessons.
"""

from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class MujocoSimulationResult:
    """Recorded scalar state from one single-joint simulation.

    Every array has shape ``(step_count + 1,)`` because the initial state at
    time zero is included before the first call to ``mj_step``.
    """

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    controls: np.ndarray


def load_mujoco_model(xml_path: str | Path) -> mujoco.MjModel:
    """Compile an MJCF XML file into MuJoCo's immutable model structure."""
    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"MuJoCo XML file does not exist: {path}")
    return mujoco.MjModel.from_xml_path(str(path))


def simulate_constant_control(
    model: mujoco.MjModel,
    *,
    joint_name: str,
    actuator_name: str,
    control: float,
    step_count: int,
) -> MujocoSimulationResult:
    """Apply one constant motor command and record position and velocity."""
    _validate_simulation_arguments(
        joint_name=joint_name,
        actuator_name=actuator_name,
        control=control,
        step_count=step_count,
    )

    joint_id = _name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    actuator_id = _name_to_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )
    _require_single_dof_joint(model, joint_id, joint_name)

    # qpos and qvel do not always use the same indices in complex robots.
    # MuJoCo stores the correct starting address for each joint in the model.
    qpos_address = int(model.jnt_qposadr[joint_id])
    qvel_address = int(model.jnt_dofadr[joint_id])
    data = mujoco.MjData(model)
    data.ctrl[actuator_id] = float(control)

    sample_count = step_count + 1
    times = np.empty(sample_count, dtype=np.float64)
    positions = np.empty(sample_count, dtype=np.float64)
    velocities = np.empty(sample_count, dtype=np.float64)
    controls = np.full(sample_count, float(control), dtype=np.float64)

    for sample_index in range(sample_count):
        # Record first, so row zero represents the untouched initial state.
        times[sample_index] = data.time
        positions[sample_index] = data.qpos[qpos_address]
        velocities[sample_index] = data.qvel[qvel_address]

        if sample_index < step_count:
            mujoco.mj_step(model, data)

    return MujocoSimulationResult(
        times=times,
        positions=positions,
        velocities=velocities,
        controls=controls,
    )


def _name_to_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    object_name: str,
) -> int:
    """Resolve a readable MJCF name and fail clearly when it is missing."""
    object_id = int(mujoco.mj_name2id(model, object_type, object_name))
    if object_id == -1:
        raise ValueError(f"MuJoCo object does not exist: {object_name}")
    return object_id


def _require_single_dof_joint(
    model: mujoco.MjModel,
    joint_id: int,
    joint_name: str,
) -> None:
    """Limit this introductory utility to hinge and slide joints."""
    joint_type = int(model.jnt_type[joint_id])
    supported_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }
    if joint_type not in supported_types:
        raise ValueError(
            f"joint must have one degree of freedom: {joint_name}"
        )


def _validate_simulation_arguments(
    joint_name: str,
    actuator_name: str,
    control: float,
    step_count: int,
) -> None:
    """Reject malformed inputs before allocating simulation arrays."""
    if not isinstance(joint_name, str) or not joint_name:
        raise ValueError("joint_name must be a non-empty string")
    if not isinstance(actuator_name, str) or not actuator_name:
        raise ValueError("actuator_name must be a non-empty string")
    if not isinstance(control, Real) or isinstance(control, bool):
        raise ValueError("control must be a real number")
    if not np.isfinite(float(control)):
        raise ValueError("control must be finite")
    if type(step_count) is not int or step_count <= 0:
        raise ValueError("step_count must be a positive integer")
