"""A readable Gymnasium reach environment backed by the MuJoCo point robot.

Environment contract:

- observation: ``[position, velocity, target_position, target_error]``
- action: one motor command in the actuator control range
- reward: negative target distance with a small action penalty
- terminated: target reached with sufficiently low velocity
- truncated: maximum environment step count reached
"""

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from robot_learning.mujoco_basics import load_mujoco_model


class PointRobotReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """Move a one-dimensional MuJoCo robot to a sampled target position."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str | Path,
        *,
        frame_skip: int = 5,
        max_episode_steps: int = 200,
        minimum_target_distance: float = 0.5,
        maximum_target_distance: float = 2.0,
        success_tolerance: float = 0.05,
        velocity_tolerance: float = 0.10,
        action_penalty_weight: float = 0.01,
        maximum_action_delay_steps: int = 20,
        maximum_observation_position_bias: float = 1.0,
    ) -> None:
        super().__init__()
        _validate_environment_settings(
            frame_skip=frame_skip,
            max_episode_steps=max_episode_steps,
            minimum_target_distance=minimum_target_distance,
            maximum_target_distance=maximum_target_distance,
            success_tolerance=success_tolerance,
            velocity_tolerance=velocity_tolerance,
            action_penalty_weight=action_penalty_weight,
            maximum_action_delay_steps=maximum_action_delay_steps,
            maximum_observation_position_bias=(
                maximum_observation_position_bias
            ),
        )

        self.model = load_mujoco_model(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.minimum_target_distance = float(minimum_target_distance)
        self.maximum_target_distance = float(maximum_target_distance)
        self.success_tolerance = float(success_tolerance)
        self.velocity_tolerance = float(velocity_tolerance)
        self.action_penalty_weight = float(action_penalty_weight)
        self.maximum_action_delay_steps = maximum_action_delay_steps
        self.maximum_observation_position_bias = float(
            maximum_observation_position_bias
        )

        self._joint_id = self._resolve_id(
            mujoco.mjtObj.mjOBJ_JOINT,
            "x_slide",
        )
        self._actuator_id = self._resolve_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "x_motor",
        )
        self._body_id = self._resolve_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "point_robot",
        )
        self._qpos_address = int(self.model.jnt_qposadr[self._joint_id])
        self._qvel_address = int(self.model.jnt_dofadr[self._joint_id])
        self.nominal_body_mass = float(self.model.body_mass[self._body_id])
        self._nominal_body_inertia = self.model.body_inertia[
            self._body_id
        ].copy()
        self.nominal_joint_damping = float(
            self.model.dof_damping[self._qvel_address]
        )
        self._elapsed_steps = 0
        self._target_position = 0.0
        self._action_gain = 1.0
        self._action_delay_steps = 0
        self._observation_position_bias = 0.0
        self._delayed_actions: deque[float] = deque()

        # Derive action bounds from MJCF instead of duplicating [-1, 1].
        actuator_range = self.model.actuator_ctrlrange[self._actuator_id]
        self.action_space = spaces.Box(
            low=np.asarray([actuator_range[0]], dtype=np.float32),
            high=np.asarray([actuator_range[1]], dtype=np.float32),
            dtype=np.float32,
        )

        # MuJoCo joint limits are soft constraints, not hard bounds on qpos.
        # Keep measured state/error unbounded instead of clipping physical data
        # to make it fit a Box. The sampled target has a genuine finite bound.
        self.observation_space = spaces.Box(
            low=np.asarray(
                [
                    -np.inf,
                    -np.inf,
                    -self.maximum_target_distance,
                    -np.inf,
                ],
                dtype=np.float32,
            ),
            high=np.asarray(
                [
                    np.inf,
                    np.inf,
                    self.maximum_target_distance,
                    np.inf,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    @property
    def control_timestep(self) -> float:
        """Return simulated seconds advanced by one environment ``step``."""
        return float(self.model.opt.timestep * self.frame_skip)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset MuJoCo state and sample or explicitly set a new target."""
        options = {} if options is None else dict(options)

        # Domain parameters are sampled or selected once per Episode.  Always
        # falling back to nominal values prevents one reset leaking into the next.
        body_mass = float(
            options.get("body_mass", self.nominal_body_mass)
        )
        joint_damping = float(
            options.get("joint_damping", self.nominal_joint_damping)
        )
        self._validate_domain_parameters(body_mass, joint_damping)

        # Deployment mismatches also reset every Episode.  The delay queue is
        # prefilled with zeros so a delay of N means exactly N inactive steps.
        action_gain = float(options.get("action_gain", 1.0))
        action_delay_steps = options.get("action_delay_steps", 0)
        observation_position_bias = float(
            options.get("observation_position_bias", 0.0)
        )
        self._validate_deployment_mismatches(
            action_gain=action_gain,
            action_delay_steps=action_delay_steps,
            observation_position_bias=observation_position_bias,
        )

        initial_position = float(options.get("initial_position", 0.0))
        # Validate explicit options before changing physics, the delay queue,
        # or the RNG. A rejected reset must leave the current Episode usable.
        target_position = float(options.get("target_position", 0.0))
        self._validate_reset_positions(initial_position, target_position)
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._elapsed_steps = 0
        self._apply_domain_parameters(body_mass, joint_damping)
        self._action_gain = action_gain
        self._action_delay_steps = action_delay_steps
        self._observation_position_bias = observation_position_bias
        self._delayed_actions.clear()
        self._delayed_actions.extend([0.0] * action_delay_steps)
        if "target_position" in options:
            target_position = float(options["target_position"])
        else:
            target_position = self._sample_target_position()
        self._validate_reset_positions(initial_position, target_position)

        self.data.qpos[self._qpos_address] = initial_position
        self.data.qvel[self._qvel_address] = 0.0
        self._target_position = target_position
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_observation()
        return observation, self._get_info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Clip one action, advance physics, and evaluate the new state."""
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"action must have shape {self.action_space.shape}"
            )
        if not np.all(np.isfinite(action_array)):
            raise ValueError("action must contain only finite values")
        clipped_action = np.clip(
            action_array,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)
        commanded_action = float(clipped_action[0])
        self._delayed_actions.append(commanded_action)
        delayed_action = self._delayed_actions.popleft()
        executed_action = float(
            np.clip(
                delayed_action * self._action_gain,
                float(self.action_space.low[0]),
                float(self.action_space.high[0]),
            )
        )
        self.data.ctrl[self._actuator_id] = executed_action

        # One policy action remains fixed for several smaller physics steps.
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._elapsed_steps += 1

        observation = self._get_observation()
        info = self._get_info()
        info.update(
            {
                "commanded_action": commanded_action,
                "delayed_action": delayed_action,
                "executed_action": executed_action,
            }
        )
        reward = -info["distance"] - self.action_penalty_weight * float(
            commanded_action ** 2
        )
        terminated = bool(info["is_success"])
        truncated = bool(
            self._elapsed_steps >= self.max_episode_steps and not terminated
        )
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Expose only the task-relevant subset of simulator state."""
        true_position = float(self.data.qpos[self._qpos_address])
        measured_position = true_position + self._observation_position_bias
        velocity = float(self.data.qvel[self._qvel_address])
        target_error = self._target_position - measured_position
        return np.asarray(
            [
                measured_position,
                velocity,
                self._target_position,
                target_error,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        """Return diagnostics that are useful but not required by the policy."""
        position = float(self.data.qpos[self._qpos_address])
        velocity = float(self.data.qvel[self._qvel_address])
        distance = abs(self._target_position - position)
        is_success = (
            distance <= self.success_tolerance
            and abs(velocity) <= self.velocity_tolerance
        )
        return {
            "position": position,
            "velocity": velocity,
            "target_position": self._target_position,
            "distance": distance,
            "is_success": is_success,
            "elapsed_steps": self._elapsed_steps,
            "body_mass": float(self.model.body_mass[self._body_id]),
            "joint_damping": float(
                self.model.dof_damping[self._qvel_address]
            ),
            "observed_position": position + self._observation_position_bias,
            "action_gain": self._action_gain,
            "action_delay_steps": self._action_delay_steps,
            "observation_position_bias": self._observation_position_bias,
        }

    def _validate_deployment_mismatches(
        self,
        *,
        action_gain: float,
        action_delay_steps: Any,
        observation_position_bias: float,
    ) -> None:
        """Validate deployment options without changing Episode state."""
        if not np.isfinite(action_gain) or action_gain <= 0.0:
            raise ValueError("action_gain must be a positive finite number")
        if type(action_delay_steps) is not int or not (
            0 <= action_delay_steps <= self.maximum_action_delay_steps
        ):
            raise ValueError(
                "action_delay_steps must be an integer between 0 and "
                f"{self.maximum_action_delay_steps}"
            )
        if (
            not np.isfinite(observation_position_bias)
            or abs(observation_position_bias)
            > self.maximum_observation_position_bias
        ):
            raise ValueError(
                "observation_position_bias must be finite and within "
                f"[-{self.maximum_observation_position_bias}, "
                f"{self.maximum_observation_position_bias}]"
            )

    @staticmethod
    def _validate_domain_parameters(
        body_mass: float,
        joint_damping: float,
    ) -> None:
        """Reject invalid dynamics before touching the model or simulation."""
        if not np.isfinite(body_mass) or body_mass <= 0.0:
            raise ValueError("body_mass must be a positive finite number")
        if not np.isfinite(joint_damping) or joint_damping < 0.0:
            raise ValueError("joint_damping must be a non-negative finite number")

    def _apply_domain_parameters(
        self,
        body_mass: float,
        joint_damping: float,
    ) -> None:
        """Apply validated mass and damping, scaling inertia with mass."""
        mass_scale = body_mass / self.nominal_body_mass
        self.model.body_mass[self._body_id] = body_mass
        # A uniformly denser sphere scales mass and rotational inertia together.
        self.model.body_inertia[self._body_id] = (
            self._nominal_body_inertia * mass_scale
        )
        self.model.dof_damping[self._qvel_address] = joint_damping
        # Recompute model constants that depend on mass before the next rollout.
        mujoco.mj_setConst(self.model, self.data)

    def _sample_target_position(self) -> float:
        """Sample left or right targets while avoiding trivial near-zero goals."""
        magnitude = self.np_random.uniform(
            self.minimum_target_distance,
            self.maximum_target_distance,
        )
        direction = self.np_random.choice(np.asarray([-1.0, 1.0]))
        return float(direction * magnitude)

    def _validate_reset_positions(
        self,
        initial_position: float,
        target_position: float,
    ) -> None:
        """Keep explicit reset options inside model and task boundaries."""
        if not np.isfinite(initial_position) or not np.isfinite(target_position):
            raise ValueError("initial_position and target_position must be finite")
        joint_range = self.model.jnt_range[self._joint_id]
        if not float(joint_range[0]) <= initial_position <= float(
            joint_range[1]
        ):
            raise ValueError("initial_position is outside the joint range")
        if abs(target_position) > self.maximum_target_distance:
            raise ValueError("target_position is outside the task range")

    def _resolve_id(
        self,
        object_type: mujoco.mjtObj,
        object_name: str,
    ) -> int:
        """Resolve one required MJCF object by name."""
        object_id = int(
            mujoco.mj_name2id(self.model, object_type, object_name)
        )
        if object_id == -1:
            raise ValueError(f"required MuJoCo object is missing: {object_name}")
        return object_id


def _validate_environment_settings(
    *,
    frame_skip: int,
    max_episode_steps: int,
    minimum_target_distance: float,
    maximum_target_distance: float,
    success_tolerance: float,
    velocity_tolerance: float,
    action_penalty_weight: float,
    maximum_action_delay_steps: int,
    maximum_observation_position_bias: float,
) -> None:
    """Reject invalid task settings before model construction."""
    if type(frame_skip) is not int or frame_skip <= 0:
        raise ValueError("frame_skip must be a positive integer")
    if type(max_episode_steps) is not int or max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be a positive integer")
    for name, value in (
        ("minimum_target_distance", minimum_target_distance),
        ("maximum_target_distance", maximum_target_distance),
        ("success_tolerance", success_tolerance),
        ("velocity_tolerance", velocity_tolerance),
        ("action_penalty_weight", action_penalty_weight),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.0 < minimum_target_distance <= maximum_target_distance:
        raise ValueError("target distance range must be positive and ordered")
    if success_tolerance <= 0.0:
        raise ValueError("success_tolerance must be positive")
    if velocity_tolerance <= 0.0:
        raise ValueError("velocity_tolerance must be positive")
    if action_penalty_weight < 0.0:
        raise ValueError("action_penalty_weight must be non-negative")
    if (
        type(maximum_action_delay_steps) is not int
        or maximum_action_delay_steps < 0
    ):
        raise ValueError("maximum_action_delay_steps must be non-negative")
    if (
        not np.isfinite(maximum_observation_position_bias)
        or maximum_observation_position_bias < 0.0
    ):
        raise ValueError(
            "maximum_observation_position_bias must be non-negative and finite"
        )
