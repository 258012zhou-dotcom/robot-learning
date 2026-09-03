"""Select PD controllers on nominal or randomized dynamics and evaluate both."""

import csv
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from robot_learning.gymnasium_rollout import (
    EpisodeResult,
    ProportionalDerivativeReachPolicy,
    calculate_position_overshoot,
    run_episode,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "021_domain_randomization.json"
MODEL_PATH = (
    PROJECT_ROOT / "experiments" / "017_mujoco_step" / "point_robot.xml"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "021_domain_randomization"


@dataclass(frozen=True)
class ControllerParameters:
    """One candidate PD controller."""

    proportional_gain: float
    derivative_gain: float


@dataclass(frozen=True)
class DomainEpisode:
    """One task seed paired with one fixed Episode-level physical domain."""

    task_seed: int
    domain_stream_seed: int | None
    domain_sample_index: int
    body_mass: float
    joint_damping: float


@dataclass(frozen=True)
class MeasuredRollout:
    """A rollout plus metrics needed for controller selection and analysis."""

    domain: DomainEpisode
    result: EpisodeResult
    duration_seconds: float
    position_overshoot: float
    final_distance: float


def load_config() -> dict[str, Any]:
    """Read experiment settings and reject empty controller searches."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    if not config["candidate_proportional_gains"]:
        raise ValueError("candidate_proportional_gains must not be empty")
    if not config["candidate_derivative_gains"]:
        raise ValueError("candidate_derivative_gains must not be empty")
    return config


def make_environment(config: dict[str, Any]) -> PointRobotReachEnv:
    """Build one environment whose reset controls Episode domain parameters."""
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


def make_nominal_domains(
    config: dict[str, Any],
    *,
    count: int,
    task_seed_start: int,
) -> list[DomainEpisode]:
    """Create fixed nominal dynamics while varying only task targets."""
    nominal = config["nominal"]
    return [
        DomainEpisode(
            task_seed=task_seed_start + index,
            domain_stream_seed=None,
            domain_sample_index=index,
            body_mass=float(nominal["body_mass"]),
            joint_damping=float(nominal["joint_damping"]),
        )
        for index in range(count)
    ]


def sample_domains(
    parameter_range: dict[str, list[float]],
    *,
    count: int,
    task_seed_start: int,
    domain_stream_seed: int,
) -> list[DomainEpisode]:
    """Sample reproducible mass and damping pairs from independent domain RNG."""
    mass_low, mass_high = map(float, parameter_range["body_mass"])
    damping_low, damping_high = map(
        float,
        parameter_range["joint_damping"],
    )
    if not 0.0 < mass_low <= mass_high:
        raise ValueError("body_mass range must be positive and ordered")
    if not 0.0 <= damping_low <= damping_high:
        raise ValueError("joint_damping range must be non-negative and ordered")

    rng = np.random.default_rng(domain_stream_seed)
    masses = rng.uniform(mass_low, mass_high, size=count)
    dampings = rng.uniform(damping_low, damping_high, size=count)
    return [
        DomainEpisode(
            task_seed=task_seed_start + index,
            domain_stream_seed=domain_stream_seed,
            domain_sample_index=index,
            body_mass=float(masses[index]),
            joint_damping=float(dampings[index]),
        )
        for index in range(count)
    ]


def candidate_controllers(
    config: dict[str, Any],
) -> list[ControllerParameters]:
    """Create the Cartesian product of Kp and Kd candidate values."""
    return [
        ControllerParameters(float(kp), float(kd))
        for kp in config["candidate_proportional_gains"]
        for kd in config["candidate_derivative_gains"]
    ]


def evaluate_controller(
    environment: PointRobotReachEnv,
    controller: ControllerParameters,
    domains: list[DomainEpisode],
) -> list[MeasuredRollout]:
    """Evaluate one unchanged controller on a prescribed domain list."""
    policy = ProportionalDerivativeReachPolicy(
        controller.proportional_gain,
        controller.derivative_gain,
        environment.action_space,
    )
    measured_rollouts: list[MeasuredRollout] = []
    for domain in domains:
        result = run_episode(
            environment,
            policy,
            seed=domain.task_seed,
            reset_options={
                "body_mass": domain.body_mass,
                "joint_damping": domain.joint_damping,
            },
        )
        measured_rollouts.append(
            MeasuredRollout(
                domain=domain,
                result=result,
                duration_seconds=(
                    result.step_count * environment.control_timestep
                ),
                position_overshoot=calculate_position_overshoot(
                    result.observations
                ),
                final_distance=abs(float(result.observations[-1, 3])),
            )
        )
    return measured_rollouts


def summarize_rollouts(
    measured_rollouts: list[MeasuredRollout],
) -> dict[str, int | float]:
    """Aggregate task quality while retaining worst observed overshoot."""
    if not measured_rollouts:
        raise ValueError("measured_rollouts must not be empty")
    successes = sum(item.result.is_success for item in measured_rollouts)
    return {
        "episode_count": len(measured_rollouts),
        "success_count": successes,
        "success_rate": successes / len(measured_rollouts),
        "mean_duration_seconds": float(
            np.mean([item.duration_seconds for item in measured_rollouts])
        ),
        "mean_position_overshoot": float(
            np.mean([item.position_overshoot for item in measured_rollouts])
        ),
        "maximum_position_overshoot": float(
            np.max([item.position_overshoot for item in measured_rollouts])
        ),
        "mean_final_distance": float(
            np.mean([item.final_distance for item in measured_rollouts])
        ),
    }


def controller_score(
    summary: dict[str, int | float],
    score_config: dict[str, float],
) -> float:
    """Combine explicit selection priorities into one lower-is-better score."""
    return float(
        float(score_config["failure_penalty"])
        * (1.0 - float(summary["success_rate"]))
        + float(score_config["duration_weight"])
        * float(summary["mean_duration_seconds"])
        + float(score_config["overshoot_weight"])
        * float(summary["mean_position_overshoot"])
        + float(score_config["final_distance_weight"])
        * float(summary["mean_final_distance"])
    )


def select_controller(
    environment: PointRobotReachEnv,
    candidates: list[ControllerParameters],
    domains: list[DomainEpisode],
    score_config: dict[str, float],
) -> tuple[ControllerParameters, list[dict[str, Any]]]:
    """Evaluate every candidate and select the minimum predefined score."""
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        summary = summarize_rollouts(
            evaluate_controller(environment, candidate, domains)
        )
        records.append(
            {
                **asdict(candidate),
                **summary,
                "selection_score": controller_score(summary, score_config),
            }
        )
    best_record = min(records, key=lambda record: record["selection_score"])
    selected = ControllerParameters(
        proportional_gain=float(best_record["proportional_gain"]),
        derivative_gain=float(best_record["derivative_gain"]),
    )
    return selected, records


def save_selection_table(
    records_by_distribution: dict[str, list[dict[str, Any]]],
) -> None:
    """Save every candidate score so selection is inspectable, not hidden."""
    fieldnames = [
        "selection_distribution",
        "proportional_gain",
        "derivative_gain",
        "episode_count",
        "success_count",
        "success_rate",
        "mean_duration_seconds",
        "mean_position_overshoot",
        "maximum_position_overshoot",
        "mean_final_distance",
        "selection_score",
    ]
    with (OUTPUT_DIR / "controller_selection.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for distribution, records in records_by_distribution.items():
            for record in records:
                writer.writerow(
                    {"selection_distribution": distribution, **record}
                )


def save_evaluation_table(
    evaluated: dict[str, dict[str, list[MeasuredRollout]]],
) -> None:
    """Save per-Episode domain parameters and outcomes for failure analysis."""
    with (OUTPUT_DIR / "evaluation_episodes.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "evaluation_distribution",
                "controller",
                "task_seed",
                "domain_stream_seed",
                "domain_sample_index",
                "body_mass",
                "joint_damping",
                "success",
                "duration_seconds",
                "position_overshoot",
                "final_distance",
            )
        )
        for distribution, controller_results in evaluated.items():
            for controller_name, measured_rollouts in controller_results.items():
                for item in measured_rollouts:
                    writer.writerow(
                        (
                            distribution,
                            controller_name,
                            item.domain.task_seed,
                            item.domain.domain_stream_seed,
                            item.domain.domain_sample_index,
                            item.domain.body_mass,
                            item.domain.joint_damping,
                            item.result.is_success,
                            item.duration_seconds,
                            item.position_overshoot,
                            item.final_distance,
                        )
                    )


def save_evaluation_plot(
    summaries: dict[str, dict[str, dict[str, int | float]]],
) -> None:
    """Compare both selected controllers across all evaluation distributions."""
    distributions = list(summaries)
    controller_names = list(next(iter(summaries.values())))
    x_positions = np.arange(len(distributions))
    width = 0.36
    metrics = (
        ("success_rate", "success rate"),
        ("mean_duration_seconds", "mean duration [s]"),
        ("mean_position_overshoot", "mean overshoot [m]"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for controller_index, controller_name in enumerate(controller_names):
        offset = (controller_index - 0.5) * width
        for axis, (metric_key, title) in zip(axes, metrics):
            values = [
                float(summaries[distribution][controller_name][metric_key])
                for distribution in distributions
            ]
            axis.bar(
                x_positions + offset,
                values,
                width,
                label=controller_name,
            )
            axis.set_title(title)
    for axis in axes:
        axis.set_xticks(x_positions, distributions, rotation=15)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axes[0].set_ylim(0.0, 1.05)
    figure.suptitle("Nominal-selected versus randomized-selected controller")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "domain_evaluation.png", dpi=160)
    plt.close(figure)


def save_ood_example_plot(
    evaluated: dict[str, dict[str, list[MeasuredRollout]]],
    environment: PointRobotReachEnv,
) -> None:
    """Plot both selected controllers on the first identical OOD task."""
    ood_results = evaluated["ood"]
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    target = float(
        next(iter(ood_results.values()))[0].result.observations[0, 2]
    )
    for controller_name, measured_rollouts in ood_results.items():
        result = measured_rollouts[0].result
        observation_times = (
            np.arange(result.observations.shape[0])
            * environment.control_timestep
        )
        action_times = (
            np.arange(result.actions.shape[0]) * environment.control_timestep
        )
        axes[0].plot(
            observation_times,
            result.observations[:, 0],
            label=controller_name,
        )
        axes[1].step(
            action_times,
            result.actions[:, 0],
            where="post",
            label=controller_name,
        )
    axes[0].axhline(target, color="black", linestyle="--", label="target")
    axes[0].set_ylabel("position [m]")
    axes[1].set(xlabel="simulation time [s]", ylabel="action")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Selected controllers on the same OOD dynamics and target")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "ood_example.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Select controllers without evaluation leakage, then compare domains."""
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

    selection_count = int(config["selection_episode_count"])
    selection_task_seed = int(config["selection_task_seed_start"])
    nominal_selection_domains = make_nominal_domains(
        config,
        count=selection_count,
        task_seed_start=selection_task_seed,
    )
    randomized_selection_domains = sample_domains(
        config["randomized_range"],
        count=selection_count,
        task_seed_start=selection_task_seed,
        domain_stream_seed=int(config["selection_domain_seed"]),
    )
    candidates = candidate_controllers(config)
    nominal_selected, nominal_records = select_controller(
        environment,
        candidates,
        nominal_selection_domains,
        config["selection_score"],
    )
    randomized_selected, randomized_records = select_controller(
        environment,
        candidates,
        randomized_selection_domains,
        config["selection_score"],
    )
    save_selection_table(
        {
            "nominal": nominal_records,
            "randomized": randomized_records,
        }
    )

    evaluation_count = int(config["evaluation_episode_count"])
    evaluation_task_seed = int(config["evaluation_task_seed_start"])
    evaluation_domains = {
        "nominal": make_nominal_domains(
            config,
            count=evaluation_count,
            task_seed_start=evaluation_task_seed,
        ),
        "in_distribution": sample_domains(
            config["randomized_range"],
            count=evaluation_count,
            task_seed_start=evaluation_task_seed,
            domain_stream_seed=int(config["in_distribution_domain_seed"]),
        ),
        "ood": sample_domains(
            config["ood_range"],
            count=evaluation_count,
            task_seed_start=evaluation_task_seed,
            domain_stream_seed=int(config["ood_domain_seed"]),
        ),
    }
    selected_controllers = {
        "nominal_selected": nominal_selected,
        "randomized_selected": randomized_selected,
    }
    evaluated = {
        distribution: {
            controller_name: evaluate_controller(
                environment,
                controller,
                domains,
            )
            for controller_name, controller in selected_controllers.items()
        }
        for distribution, domains in evaluation_domains.items()
    }
    evaluation_summaries = {
        distribution: {
            controller_name: summarize_rollouts(measured_rollouts)
            for controller_name, measured_rollouts in controller_results.items()
        }
        for distribution, controller_results in evaluated.items()
    }

    save_evaluation_table(evaluated)
    save_evaluation_plot(evaluation_summaries)
    save_ood_example_plot(evaluated, environment)
    output = {
        "experiment_name": config["experiment_name"],
        "selection_is_controller_tuning_not_rl_training": True,
        "selection_score": config["selection_score"],
        "selected_controllers": {
            name: asdict(controller)
            for name, controller in selected_controllers.items()
        },
        "domain_definitions": {
            "nominal": config["nominal"],
            "in_distribution": config["randomized_range"],
            "ood": config["ood_range"],
        },
        "evaluation_summaries": evaluation_summaries,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    logging.info("nominal 选择控制器：%s", nominal_selected)
    logging.info("randomized 选择控制器：%s", randomized_selected)
    for distribution, controller_results in evaluation_summaries.items():
        for controller_name, summary in controller_results.items():
            logging.info(
                "%s / %s：成功率 %.1f%%，时间 %.3f s，超调 %.4f m",
                distribution,
                controller_name,
                100.0 * summary["success_rate"],
                summary["mean_duration_seconds"],
                summary["mean_position_overshoot"],
            )
    logging.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
