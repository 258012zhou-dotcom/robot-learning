"""Calibrate hidden deployment mismatch and evaluate transfer on new tasks."""

import csv
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from robot_learning.gymnasium_rollout import (
    Policy,
    ProportionalDerivativeReachPolicy,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.sim_to_sim_calibration import (
    CalibratedPDReachPolicy,
    collect_step_response,
    DeploymentCalibration,
    estimate_deployment_calibration,
    StepResponseTrace,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "022_sim_to_sim_calibration.json"
MODEL_PATH = (
    PROJECT_ROOT / "experiments" / "017_mujoco_step" / "point_robot.xml"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "022_sim_to_sim_calibration"


@dataclass(frozen=True)
class TransferRollout:
    """True and measured evidence from one held-out deployment Episode."""

    policy_name: str
    seed: int
    target_position: float
    true_positions: np.ndarray
    measured_positions: np.ndarray
    commanded_actions: np.ndarray
    executed_actions: np.ndarray
    success: bool
    duration_seconds: float
    final_true_distance: float
    true_position_overshoot: float


def load_config() -> dict[str, Any]:
    """Load the fixed calibration and evaluation protocol."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def make_environment(config: dict[str, Any]) -> PointRobotReachEnv:
    """Build one environment shared by calibration and deployment evaluation."""
    return PointRobotReachEnv(
        MODEL_PATH,
        frame_skip=int(config["frame_skip"]),
        max_episode_steps=int(config["max_episode_steps"]),
        minimum_target_distance=float(config["minimum_target_distance"]),
        maximum_target_distance=float(config["maximum_target_distance"]),
        success_tolerance=float(config["success_tolerance"]),
        velocity_tolerance=float(config["velocity_tolerance"]),
        action_penalty_weight=float(config["action_penalty_weight"]),
    )


def deployment_options(config: dict[str, Any]) -> dict[str, int | float]:
    """Return a copy of the hidden target-domain settings used by the harness."""
    domain = config["hidden_deployment_domain"]
    return {
        "action_gain": float(domain["action_gain"]),
        "action_delay_steps": int(domain["action_delay_steps"]),
        "observation_position_bias": float(
            domain["observation_position_bias"]
        ),
    }


def measure_nominal_step_velocity(
    environment: PointRobotReachEnv,
    *,
    command: float,
    initial_position: float,
    target_position: float,
    seed: int,
) -> float:
    """Measure the source simulator's first-step velocity for a known command."""
    environment.reset(
        seed=seed,
        options={
            "initial_position": initial_position,
            "target_position": target_position,
        },
    )
    observation, _, _, _, _ = environment.step(
        np.asarray([command], dtype=np.float32)
    )
    return float(observation[1])


def calibrate_target_domain(
    config: dict[str, Any],
    environment: PointRobotReachEnv,
) -> tuple[DeploymentCalibration, StepResponseTrace, float]:
    """Collect one target trace and estimate mismatch without using true info."""
    calibration_config = config["calibration"]
    initial_position = float(calibration_config["known_initial_position"])
    target_position = float(calibration_config["target_position"])
    command = float(calibration_config["command"])
    seed = int(calibration_config["seed"])

    # This nominal response is our simple source-model prediction.  A real
    # project would derive it from the calibrated simulator or dynamics model.
    nominal_velocity = measure_nominal_step_velocity(
        environment,
        command=command,
        initial_position=initial_position,
        target_position=target_position,
        seed=seed,
    )
    trace = collect_step_response(
        environment,
        seed=seed,
        reset_options={
            "initial_position": initial_position,
            "target_position": target_position,
            **deployment_options(config),
        },
        command=command,
        step_count=int(calibration_config["step_count"]),
    )
    estimate = estimate_deployment_calibration(
        trace,
        known_initial_position=initial_position,
        nominal_step_velocity=nominal_velocity,
        velocity_threshold=float(calibration_config["velocity_threshold"]),
    )
    return estimate, trace, nominal_velocity


def make_policies(
    config: dict[str, Any],
    environment: PointRobotReachEnv,
    estimate: DeploymentCalibration,
) -> dict[str, Policy]:
    """Create direct, estimated-calibration, and oracle comparison policies."""
    controller = config["controller"]
    proportional_gain = float(controller["proportional_gain"])
    derivative_gain = float(controller["derivative_gain"])
    shared = {
        "proportional_gain": proportional_gain,
        "derivative_gain": derivative_gain,
        "action_space": environment.action_space,
    }
    true_domain = deployment_options(config)
    oracle = DeploymentCalibration(
        action_gain=float(true_domain["action_gain"]),
        action_delay_steps=int(true_domain["action_delay_steps"]),
        observation_position_bias=float(
            true_domain["observation_position_bias"]
        ),
    )
    return {
        "direct_nominal": ProportionalDerivativeReachPolicy(**shared),
        "estimated_calibration": CalibratedPDReachPolicy(
            **shared,
            control_timestep=environment.control_timestep,
            calibration=estimate,
        ),
        "oracle_calibration": CalibratedPDReachPolicy(
            **shared,
            control_timestep=environment.control_timestep,
            calibration=oracle,
        ),
    }


def calculate_overshoot(
    true_positions: np.ndarray,
    target_position: float,
) -> float:
    """Measure overshoot from true positions, independent of sensor bias."""
    initial_position = float(true_positions[0])
    displacement = target_position - initial_position
    if displacement == 0.0:
        return 0.0
    direction = float(np.sign(displacement))
    progress = direction * (true_positions - initial_position)
    return max(0.0, float(np.max(progress) - abs(displacement)))


def run_transfer_episode(
    environment: PointRobotReachEnv,
    policy: Policy,
    *,
    policy_name: str,
    seed: int,
    target_options: dict[str, int | float],
) -> TransferRollout:
    """Run one target-domain Episode while keeping truth out of the policy."""
    observation, info = environment.reset(
        seed=seed,
        options={"initial_position": 0.0, **target_options},
    )
    target_position = float(info["target_position"])
    true_positions = [float(info["position"])]
    measured_positions = [float(observation[0])]
    commanded_actions: list[float] = []
    executed_actions: list[float] = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = np.asarray(policy(observation), dtype=np.float32)
        observation, _, terminated, truncated, info = environment.step(action)
        true_positions.append(float(info["position"]))
        measured_positions.append(float(observation[0]))
        commanded_actions.append(float(info["commanded_action"]))
        executed_actions.append(float(info["executed_action"]))

    true_position_array = np.asarray(true_positions, dtype=np.float64)
    return TransferRollout(
        policy_name=policy_name,
        seed=seed,
        target_position=target_position,
        true_positions=true_position_array,
        measured_positions=np.asarray(measured_positions, dtype=np.float64),
        commanded_actions=np.asarray(commanded_actions, dtype=np.float64),
        executed_actions=np.asarray(executed_actions, dtype=np.float64),
        success=bool(info["is_success"]),
        duration_seconds=(
            len(commanded_actions) * environment.control_timestep
        ),
        final_true_distance=abs(target_position - true_position_array[-1]),
        true_position_overshoot=calculate_overshoot(
            true_position_array,
            target_position,
        ),
    )


def evaluate_policies(
    config: dict[str, Any],
    environment: PointRobotReachEnv,
    policies: dict[str, Policy],
) -> dict[str, list[TransferRollout]]:
    """Evaluate frozen policies on the same held-out target seeds."""
    evaluation = config["evaluation"]
    seeds = range(
        int(evaluation["seed_start"]),
        int(evaluation["seed_start"]) + int(evaluation["episode_count"]),
    )
    target_options = deployment_options(config)
    return {
        policy_name: [
            run_transfer_episode(
                environment,
                policy,
                policy_name=policy_name,
                seed=seed,
                target_options=target_options,
            )
            for seed in seeds
        ]
        for policy_name, policy in policies.items()
    }


def summarize_rollouts(
    rollouts: list[TransferRollout],
) -> dict[str, int | float]:
    """Aggregate held-out performance using true physical state."""
    if not rollouts:
        raise ValueError("rollouts must not be empty")
    success_count = sum(rollout.success for rollout in rollouts)
    return {
        "episode_count": len(rollouts),
        "success_count": success_count,
        "success_rate": success_count / len(rollouts),
        "mean_duration_seconds": float(
            np.mean([rollout.duration_seconds for rollout in rollouts])
        ),
        "mean_final_true_distance": float(
            np.mean([rollout.final_true_distance for rollout in rollouts])
        ),
        "mean_true_position_overshoot": float(
            np.mean([rollout.true_position_overshoot for rollout in rollouts])
        ),
        "mean_absolute_command": float(
            np.mean(
                [
                    np.mean(np.abs(rollout.commanded_actions))
                    for rollout in rollouts
                ]
            )
        ),
        "mean_absolute_executed_action": float(
            np.mean(
                [
                    np.mean(np.abs(rollout.executed_actions))
                    for rollout in rollouts
                ]
            )
        ),
    }


def save_calibration_trace(
    trace: StepResponseTrace,
    estimate: DeploymentCalibration,
) -> None:
    """Save the measured calibration evidence and detected response step."""
    with (OUTPUT_DIR / "calibration_trace.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ("step_index", "commanded_action", "measured_position", "velocity")
        )
        writer.writerow((-1, 0.0, trace.initial_observation[0], 0.0))
        for step_index, (command, observation) in enumerate(
            zip(trace.commanded_actions, trace.observations)
        ):
            writer.writerow(
                (step_index, command, observation[0], observation[1])
            )

    steps = np.arange(trace.observations.shape[0])
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].step(
        steps,
        trace.commanded_actions,
        where="post",
        label="commanded action",
    )
    axes[1].plot(steps, trace.observations[:, 1], marker="o", label="velocity")
    axes[1].axvline(
        estimate.action_delay_steps,
        color="tab:red",
        linestyle="--",
        label="first detected response",
    )
    axes[0].set_ylabel("command")
    axes[1].set(xlabel="control step", ylabel="measured velocity")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Calibration step response")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "calibration_response.png", dpi=160)
    plt.close(figure)


def save_evaluation_table(
    evaluated: dict[str, list[TransferRollout]],
) -> None:
    """Preserve every held-out Episode instead of only aggregate means."""
    with (OUTPUT_DIR / "evaluation_episodes.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "policy",
                "seed",
                "target_position",
                "success",
                "duration_seconds",
                "final_true_distance",
                "true_position_overshoot",
            )
        )
        for policy_name, rollouts in evaluated.items():
            for rollout in rollouts:
                writer.writerow(
                    (
                        policy_name,
                        rollout.seed,
                        rollout.target_position,
                        rollout.success,
                        rollout.duration_seconds,
                        rollout.final_true_distance,
                        rollout.true_position_overshoot,
                    )
                )


def save_evaluation_plot(
    summaries: dict[str, dict[str, int | float]],
) -> None:
    """Plot success, duration, and true final error for all policies."""
    policy_names = list(summaries)
    x_positions = np.arange(len(policy_names))
    metrics = (
        ("success_rate", "success rate"),
        ("mean_duration_seconds", "mean duration [s]"),
        ("mean_final_true_distance", "mean final true distance [m]"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, (metric, title) in zip(axes, metrics):
        axis.bar(
            x_positions,
            [float(summaries[name][metric]) for name in policy_names],
        )
        axis.set_title(title)
        axis.set_xticks(x_positions, policy_names, rotation=18)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.0, 1.05)
    figure.suptitle("Held-out Sim-to-Sim transfer evaluation")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "transfer_evaluation.png", dpi=160)
    plt.close(figure)


def save_example_plot(
    evaluated: dict[str, list[TransferRollout]],
    control_timestep: float,
) -> None:
    """Show true trajectories for the first identical held-out target."""
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    first_rollout = next(iter(evaluated.values()))[0]
    for policy_name, rollouts in evaluated.items():
        rollout = rollouts[0]
        position_times = (
            np.arange(rollout.true_positions.size) * control_timestep
        )
        action_times = (
            np.arange(rollout.commanded_actions.size) * control_timestep
        )
        axes[0].plot(
            position_times,
            rollout.true_positions,
            label=policy_name,
        )
        axes[1].step(
            action_times,
            rollout.commanded_actions,
            where="post",
            label=policy_name,
        )
    axes[0].axhline(
        first_rollout.target_position,
        color="black",
        linestyle="--",
        label="target",
    )
    axes[0].set_ylabel("true position [m]")
    axes[1].set(xlabel="simulation time [s]", ylabel="commanded action")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Direct transfer versus calibrated compensation")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "transfer_example.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Calibrate once, freeze the estimate, and evaluate on held-out tasks."""
    config = load_config()
    environment = make_environment(config)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=(
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "run.log", encoding="utf-8"),
        ),
    )

    estimate, trace, nominal_velocity = calibrate_target_domain(
        config,
        environment,
    )
    policies = make_policies(config, environment, estimate)
    evaluated = evaluate_policies(config, environment, policies)
    summaries = {
        policy_name: summarize_rollouts(rollouts)
        for policy_name, rollouts in evaluated.items()
    }

    save_calibration_trace(trace, estimate)
    save_evaluation_table(evaluated)
    save_evaluation_plot(summaries)
    save_example_plot(evaluated, environment.control_timestep)
    true_calibration = DeploymentCalibration(
        **deployment_options(config),
    )
    results = {
        "experiment_name": config["experiment_name"],
        "transfer_type": "Sim-to-Sim proxy, not real-hardware Sim-to-Real",
        "calibration_episode_count": 1,
        "nominal_step_velocity": nominal_velocity,
        "true_hidden_parameters_revealed_after_estimation": asdict(
            true_calibration
        ),
        "estimated_parameters": asdict(estimate),
        "absolute_estimation_errors": {
            "action_gain": abs(
                estimate.action_gain - true_calibration.action_gain
            ),
            "action_delay_steps": abs(
                estimate.action_delay_steps
                - true_calibration.action_delay_steps
            ),
            "observation_position_bias": abs(
                estimate.observation_position_bias
                - true_calibration.observation_position_bias
            ),
        },
        "held_out_evaluation": summaries,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    logging.info("隐藏参数（评价时揭示）：%s", true_calibration)
    logging.info("校准估计：%s", estimate)
    for policy_name, summary in summaries.items():
        logging.info(
            "%s：成功率 %.1f%%，时间 %.3f s，最终真实误差 %.4f m",
            policy_name,
            100.0 * summary["success_rate"],
            summary["mean_duration_seconds"],
            summary["mean_final_true_distance"],
        )
    logging.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
