"""Small, inspectable tools for a Sim-to-Sim deployment calibration lesson."""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class DeploymentCalibration:
    """Estimated actuator and position-sensor mismatch."""

    action_gain: float
    action_delay_steps: int
    observation_position_bias: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.action_gain) or self.action_gain <= 0.0:
            raise ValueError("action_gain must be a positive finite number")
        if (
            type(self.action_delay_steps) is not int
            or self.action_delay_steps < 0
        ):
            raise ValueError("action_delay_steps must be a non-negative integer")
        if not np.isfinite(self.observation_position_bias):
            raise ValueError("observation_position_bias must be finite")


@dataclass(frozen=True)
class StepResponseTrace:
    """Measured observations produced by repeatedly sending one command."""

    initial_observation: np.ndarray
    commanded_actions: np.ndarray
    observations: np.ndarray


def collect_step_response(
    environment: gym.Env,
    *,
    seed: int,
    reset_options: dict[str, Any],
    command: float,
    step_count: int,
) -> StepResponseTrace:
    """Send one fixed command and record only policy-visible observations.

    The function deliberately does not read true actuator gain, delay, or sensor
    bias from ``info``.  This mirrors calibration where hidden hardware
    properties must be inferred from commands and measurements.
    """
    if not np.isfinite(command):
        raise ValueError("command must be finite")
    if type(step_count) is not int or step_count <= 0:
        raise ValueError("step_count must be a positive integer")

    initial_observation, _ = environment.reset(
        seed=seed,
        options=reset_options,
    )
    action = np.asarray([command], dtype=np.float32)
    if not environment.action_space.contains(action):
        raise ValueError("command is outside the environment action space")

    observations: list[np.ndarray] = []
    for _ in range(step_count):
        observation, _, terminated, truncated, _ = environment.step(action)
        observations.append(np.asarray(observation).copy())
        if terminated or truncated:
            raise RuntimeError("calibration Episode ended before probe completed")

    return StepResponseTrace(
        initial_observation=np.asarray(initial_observation).copy(),
        commanded_actions=np.full(step_count, command, dtype=np.float32),
        observations=np.stack(observations),
    )


def estimate_deployment_calibration(
    trace: StepResponseTrace,
    *,
    known_initial_position: float,
    nominal_step_velocity: float,
    velocity_threshold: float,
) -> DeploymentCalibration:
    """Estimate bias, delay, and gain from one known step response.

    Assumptions kept explicit for this first experiment:

    - the robot starts at a known homed position;
    - calibration and nominal simulation use the same constant command;
    - mass and damping already match, so the first velocity ratio identifies
      actuator gain;
    - position bias is constant during the Episode.
    """
    if trace.initial_observation.ndim != 1:
        raise ValueError("initial_observation must be one-dimensional")
    if trace.observations.ndim != 2 or trace.observations.shape[1] < 2:
        raise ValueError("observations must have shape (T, at_least_2)")
    if trace.commanded_actions.shape != (trace.observations.shape[0],):
        raise ValueError("one commanded action is required per observation")
    if not np.isfinite(known_initial_position):
        raise ValueError("known_initial_position must be finite")
    if not np.isfinite(nominal_step_velocity) or nominal_step_velocity == 0.0:
        raise ValueError("nominal_step_velocity must be finite and non-zero")
    if not np.isfinite(velocity_threshold) or velocity_threshold <= 0.0:
        raise ValueError("velocity_threshold must be positive and finite")

    measured_velocities = trace.observations[:, 1]
    response_indices = np.flatnonzero(
        np.abs(measured_velocities) > velocity_threshold
    )
    if response_indices.size == 0:
        raise ValueError("no actuator response was detected")

    first_response_index = int(response_indices[0])
    measured_step_velocity = float(measured_velocities[first_response_index])
    action_gain = abs(measured_step_velocity / nominal_step_velocity)
    position_bias = (
        float(trace.initial_observation[0]) - known_initial_position
    )
    return DeploymentCalibration(
        action_gain=action_gain,
        action_delay_steps=first_response_index,
        observation_position_bias=position_bias,
    )


class CalibratedPDReachPolicy:
    """Compensate measured bias, actuator gain, and a short action delay."""

    def __init__(
        self,
        *,
        proportional_gain: float,
        derivative_gain: float,
        action_space: gym.spaces.Box,
        control_timestep: float,
        calibration: DeploymentCalibration,
    ) -> None:
        if proportional_gain <= 0.0:
            raise ValueError("proportional_gain must be positive")
        if derivative_gain < 0.0:
            raise ValueError("derivative_gain must be non-negative")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("action_space must be a Box")
        if action_space.shape != (1,):
            raise ValueError("policy expects one-dimensional actions")
        if control_timestep <= 0.0:
            raise ValueError("control_timestep must be positive")

        self.proportional_gain = float(proportional_gain)
        self.derivative_gain = float(derivative_gain)
        self.control_timestep = float(control_timestep)
        self.calibration = calibration
        self._low = float(action_space.low[0])
        self._high = float(action_space.high[0])

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Estimate the current/future true state, then compensate action gain."""
        observation_array = np.asarray(observation)
        if observation_array.shape != (4,):
            raise ValueError("observation must have shape (4,)")

        measured_position = float(observation_array[0])
        velocity = float(observation_array[1])
        target_position = float(observation_array[2])

        estimated_position = (
            measured_position - self.calibration.observation_position_bias
        )
        delay_seconds = (
            self.calibration.action_delay_steps * self.control_timestep
        )
        predicted_position = estimated_position + velocity * delay_seconds
        predicted_error = target_position - predicted_position

        desired_executed_action = (
            self.proportional_gain * predicted_error
            - self.derivative_gain * velocity
        )
        commanded_action = (
            desired_executed_action / self.calibration.action_gain
        )
        clipped_action = np.clip(commanded_action, self._low, self._high)
        return np.asarray([clipped_action], dtype=np.float32)
