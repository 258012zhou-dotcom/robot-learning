"""Benchmark single, synchronous-vector, and asynchronous-vector stepping."""

import argparse
import csv
from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from robot_learning.vector_environment_benchmark import (
    BackendName,
    BenchmarkResult,
    PointRobotBenchmarkFactory,
    run_benchmark_once,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "023_vector_environment_benchmark.json"
)
MODEL_PATH = (
    PROJECT_ROOT / "experiments" / "017_mujoco_step" / "point_robot.xml"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "023_vector_environment_benchmark"
BACKENDS: tuple[BackendName, ...] = (
    "single",
    "sync_vector",
    "async_vector",
)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load benchmark settings and reject duplicate workload names."""
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    names = [workload["name"] for workload in config["workloads"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("workload names must be non-empty and unique")
    return config


def run_all_benchmarks(
    config: dict[str, Any],
) -> dict[str, dict[str, list[BenchmarkResult]]]:
    """Run every workload/backend pair with identical repeated seeds."""
    records: dict[str, dict[str, list[BenchmarkResult]]] = {}
    for workload in config["workloads"]:
        workload_name = str(workload["name"])
        transition_count = int(workload["total_transition_count"])
        warmup_steps = int(config["warmup_vector_steps"])
        factory = PointRobotBenchmarkFactory(
            xml_path=str(MODEL_PATH),
            frame_skip=int(workload["frame_skip"]),
            # Both phases start at step zero after a seeded reset.
            max_episode_steps=max(
                transition_count // int(config["vector_environment_count"]),
                warmup_steps,
            ) + 10,
        )
        records[workload_name] = {}
        for backend in BACKENDS:
            repetitions: list[BenchmarkResult] = []
            for repetition_index in range(int(config["repetition_count"])):
                logging.info(
                    "开始 %s / %s，第 %d/%d 次",
                    workload_name,
                    backend,
                    repetition_index + 1,
                    int(config["repetition_count"]),
                )
                repetitions.append(
                    run_benchmark_once(
                        backend=backend,
                        factory=factory,
                        vector_environment_count=int(
                            config["vector_environment_count"]
                        ),
                        total_transition_count=transition_count,
                        warmup_vector_steps=warmup_steps,
                        base_seed=int(config["base_seed"]),
                    )
                )
            records[workload_name][backend] = repetitions
    return records


def verify_reproducibility(
    records: dict[str, dict[str, list[BenchmarkResult]]],
) -> dict[str, dict[str, bool]]:
    """Check deterministic data separately from nondeterministic timing."""
    checks: dict[str, dict[str, bool]] = {}
    for workload_name, backend_records in records.items():
        checks[workload_name] = {}
        for backend, repetitions in backend_records.items():
            checks[workload_name][f"{backend}_repeated_trajectory_match"] = (
                bool(repetitions) and all(
                    record.trajectory_sha256 == repetitions[0].trajectory_sha256
                    for record in repetitions
                )
            )

        # Include every repetition of the serial baseline as well as vectors.
        reference = backend_records["single"][0]
        checks[workload_name]["all_backends_trajectory_match"] = all(
            record.trajectory_sha256 == reference.trajectory_sha256
            and record.environment_count == reference.environment_count
            and record.vector_step_count == reference.vector_step_count
            and record.transition_count == reference.transition_count
            for backend in BACKENDS
            for record in backend_records[backend]
        )
    return checks


def summarize_records(
    records: dict[str, dict[str, list[BenchmarkResult]]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Aggregate timing variation and calculate speedup within each workload."""
    summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    for workload_name, backend_records in records.items():
        summaries[workload_name] = {}
        single_throughputs = [
            item.transitions_per_second
            for item in backend_records["single"]
        ]
        single_median = float(np.median(single_throughputs))
        for backend, repetitions in backend_records.items():
            throughputs = np.asarray(
                [item.transitions_per_second for item in repetitions],
                dtype=np.float64,
            )
            initialization_times = np.asarray(
                [item.initialization_seconds for item in repetitions],
                dtype=np.float64,
            )
            rollout_times = np.asarray(
                [item.rollout_seconds for item in repetitions],
                dtype=np.float64,
            )
            median_throughput = float(np.median(throughputs))
            summaries[workload_name][backend] = {
                "environment_count": repetitions[0].environment_count,
                "transition_count_per_repetition": (
                    repetitions[0].transition_count
                ),
                "vector_step_count_per_repetition": (
                    repetitions[0].vector_step_count
                ),
                "mean_initialization_seconds": float(
                    np.mean(initialization_times)
                ),
                "mean_rollout_seconds": float(np.mean(rollout_times)),
                "median_transitions_per_second": median_throughput,
                "standard_deviation_transitions_per_second": float(
                    np.std(throughputs)
                ),
                "speedup_over_single": median_throughput / single_median,
            }
    return summaries


def save_raw_records(
    records: dict[str, dict[str, list[BenchmarkResult]]],
) -> None:
    """Save every repetition so aggregate performance remains auditable."""
    fieldnames = [
        "workload",
        "repetition",
        "backend",
        "environment_count",
        "transition_count",
        "vector_step_count",
        "initialization_seconds",
        "rollout_seconds",
        "transitions_per_second",
        "checksum",
        "trajectory_sha256",
        "warmup_seconds",
        "reset_seconds",
    ]
    with (OUTPUT_DIR / "benchmark_repetitions.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for workload_name, backend_records in records.items():
            for backend, repetitions in backend_records.items():
                for repetition_index, result in enumerate(repetitions):
                    writer.writerow(
                        {
                            "workload": workload_name,
                            "repetition": repetition_index,
                            **asdict(result),
                        }
                    )


def save_benchmark_plot(
    summaries: dict[str, dict[str, dict[str, float | int]]],
) -> None:
    """Compare steady-state throughput and initialization overhead."""
    workloads = list(summaries)
    x_positions = np.arange(len(workloads), dtype=np.float64)
    width = 0.24
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for backend_index, backend in enumerate(BACKENDS):
        offsets = x_positions + (backend_index - 1) * width
        throughputs = [
            float(summaries[name][backend]["median_transitions_per_second"])
            for name in workloads
        ]
        deviations = [
            float(
                summaries[name][backend][
                    "standard_deviation_transitions_per_second"
                ]
            )
            for name in workloads
        ]
        initialization_times = [
            float(summaries[name][backend]["mean_initialization_seconds"])
            for name in workloads
        ]
        axes[0].bar(
            offsets,
            throughputs,
            width,
            yerr=deviations,
            capsize=3,
            label=backend,
        )
        axes[1].bar(
            offsets,
            initialization_times,
            width,
            label=backend,
        )

    axes[0].set_ylabel("median transitions / second")
    axes[0].set_title("steady-state rollout throughput")
    axes[1].set_ylabel("mean initialization time [s]")
    axes[1].set_title("construction and first reset overhead")
    for axis in axes:
        axis.set_xticks(x_positions, workloads)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle("Serial pool versus SyncVectorEnv versus AsyncVectorEnv")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "vector_environment_benchmark.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Run the fixed benchmark protocol and report facts, not assumptions."""
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    OUTPUT_DIR = args.output_dir
    # Refuse to overwrite historical evidence, including an existing run.log.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError("output directory must be new or empty")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=(
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "run.log", encoding="utf-8"),
        ),
    )

    records = run_all_benchmarks(config)
    reproducibility = verify_reproducibility(records)
    if not all(
        passed
        for workload_checks in reproducibility.values()
        for passed in workload_checks.values()
    ):
        raise RuntimeError("a benchmark backend changed deterministic results")
    summaries = summarize_records(records)
    save_raw_records(records)
    save_benchmark_plot(summaries)

    results = {
        "experiment_name": config["experiment_name"],
        "protocol_version": 2,
        "single_means_serial_environment_pool": True,
        "measures_learning_performance": False,
        "base_seed": config["base_seed"],
        "seed_rule": "base_seed + environment_index",
        "action_rule": "float32(0.75*sin(0.017*step + 0.37*environment_index))",
        "warmup_vector_steps": config["warmup_vector_steps"],
        "timing_scope": (
            "rollout includes action generation, step, checks and SHA-256; "
            "construction/first reset, warmup, second seeded reset and close excluded"
        ),
        "trajectory_hash_scope": (
            "SHA-256: initial observations, then actions/observations/rewards/"
            "terminated/truncated in step, environment, component order; "
            "each field encoded as ndim/shape <i8 followed by C-order <f8 values"
        ),
        "timing_is_machine_and_load_dependent": True,
        "fixed_total_transitions_within_each_workload": True,
        "vector_environment_count": config["vector_environment_count"],
        "repetition_count": config["repetition_count"],
        "workloads": config["workloads"],
        "reproducibility_checks": reproducibility,
        "summaries": summaries,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    for workload_name, backend_summaries in summaries.items():
        for backend, summary in backend_summaries.items():
            logging.info(
                "%s / %s：%.0f transitions/s，相对串行池 %.2fx，初始化 %.4f s",
                workload_name,
                backend,
                summary["median_transitions_per_second"],
                summary["speedup_over_single"],
                summary["mean_initialization_seconds"],
            )
    logging.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    # AsyncVectorEnv starts worker processes.  The main guard prevents workers
    # from recursively running this whole benchmark when a spawn context is used.
    main()
