"""Replay every failed test rollout and inspect where the frozen policies fail."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from robot_learning.gymnasium_rollout import EpisodeResult, run_episode
from robot_learning.lowdim_action_diffusion import (
    DiffusionChunkPolicy,
    load_diffusion_checkpoint,
)
from robot_learning.two_route_navigation import (
    TwoRouteExpert,
    TwoRouteNavigationEnv,
    segment_clearance,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/053_diffusion_two_route_foundation"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def run_environment(task: dict) -> TwoRouteNavigationEnv:
    return TwoRouteNavigationEnv(
        dt=float(task["dt"]),
        max_episode_steps=int(task["max_episode_steps"]),
        goal_tolerance=float(task["goal_tolerance"]),
    )


def inspect_trajectory(
    result: EpisodeResult, *, execution_horizon: int
) -> dict:
    first = result.observations[0]
    obstacle_center = first[4:6]
    radius = float(first[6])
    clearances = [
        segment_clearance(start[:2], end[:2], obstacle_center, radius)
        for start, end in zip(
            result.observations[:-1], result.observations[1:], strict=True
        )
    ]
    # A sign reversal far before the left obstacle edge is descriptive, not causal.
    approach_signs = []
    for start in range(0, result.step_count, execution_horizon):
        if result.observations[start, 0] >= float(first[4] - radius):
            continue
        mean_vertical = float(result.actions[start:start + execution_horizon, 1].mean())
        if abs(mean_vertical) > 0.1:
            approach_signs.append(int(np.sign(mean_vertical)))
    return {
        "minimum_segment_clearance": float(min(clearances)),
        "closest_observed_goal_distance": float(np.min(np.linalg.norm(
            result.observations[:, :2] - first[None, 2:4], axis=1
        ))),
        "last_position": result.observations[-1, :2].tolist(),
        "obstacle_center": obstacle_center.tolist(),
        "obstacle_radius": radius,
        "goal_position": first[2:4].tolist(),
        "first_four_mean_vertical": float(
            result.actions[:execution_horizon, 1].mean()
        ),
        "early_approach_direction_reversals": sum(
            previous != current
            for previous, current in zip(approach_signs[:-1], approach_signs[1:])
        ),
    }


def replay_failures(task: dict, config: dict, source_hash: str) -> tuple[dict, dict]:
    summaries = {}
    examples = {}
    environment = run_environment(task)
    try:
        for seed in config["repeat_train_seeds"]:
            stem = "dp" if seed == int(config["seed"]) else f"dp_seed_{seed}"
            report = read_json(OUTPUT_DIR / f"{stem}_results.json")
            model, normalization, artifact = load_diffusion_checkpoint(
                OUTPUT_DIR / f"{stem}_best_model.pt"
            )
            if (report["source_dataset_sha256"] != source_hash or
                    artifact["source_dataset_sha256"] != source_hash or
                    int(artifact["seed"]) != seed):
                raise RuntimeError(f"seed {seed} report/checkpoint dataset identity mismatch")
            horizon = int(config["execution_horizon"])
            if (model.horizon != int(config["prediction_horizon"]) or
                    model.diffusion_steps != int(config["diffusion_steps"])):
                raise RuntimeError(f"seed {seed} model protocol differs")
            policy = DiffusionChunkPolicy(
                model, normalization, execution_horizon=horizon
            )
            saved_episodes = report["test_closed_loop"]["episodes"]
            records = []
            success_count = 0
            successes_with_early_reversal = 0
            for row in saved_episodes:
                policy.reset(seed=int(row["sampling_seed"]))
                result = run_episode(environment, policy, seed=int(row["scene_seed"]))
                collision = bool(result.terminated and not result.is_success)
                if (result.is_success != row["success"] or
                        collision != row["collision"] or
                        result.truncated != row["timeout"] or
                        result.step_count != row["steps"] or
                        not np.allclose(result.actions[0], row["first_action"], atol=1e-6) or
                        not np.isclose(float(np.linalg.norm(
                            result.observations[-1, :2] - result.observations[0, 2:4]
                        )), row["final_distance"], atol=1e-5)):
                    raise RuntimeError(
                        f"rollout did not replay: seed={seed}, "
                        f"scene={row['scene_seed']}, draw={row['draw']}"
                    )
                details = inspect_trajectory(result, execution_horizon=horizon)
                if result.is_success:
                    success_count += 1
                    successes_with_early_reversal += (
                        details["early_approach_direction_reversals"] > 0
                    )
                    continue
                records.append({**row, **details})
                if seed == int(config["seed"]) and row["draw"] == 0:
                    key = "collision" if collision else "timeout"
                    examples.setdefault(key, (int(row["scene_seed"]), result))
            if len(records) != (
                report["test_closed_loop"]["collision_count"]
                + report["test_closed_loop"]["timeout_count"]
            ) or success_count != report["test_closed_loop"]["success_count"]:
                raise RuntimeError("saved failure counts disagree with replay")
            collisions = [row for row in records if row["collision"]]
            timeouts = [row for row in records if row["timeout"]]
            summaries[str(seed)] = {
                "replayed_success_count": success_count,
                "successes_with_early_approach_direction_reversal": (
                    successes_with_early_reversal
                ),
                "replayed_failure_count": len(records),
                "collision_count": len(collisions),
                "collision_after_first_replan_count": sum(
                    row["steps"] > horizon for row in collisions
                ),
                "collision_step_min_max": [
                    min((row["steps"] for row in collisions), default=None),
                    max((row["steps"] for row in collisions), default=None),
                ],
                "timeout_count": len(timeouts),
                "timeout_final_distance_mean": (
                    float(np.mean([row["final_distance"] for row in timeouts]))
                    if timeouts else None
                ),
                "timeout_past_goal_x_count": sum(
                    row["last_position"][0] > row["goal_position"][0]
                    for row in timeouts
                ),
                "timeout_closest_goal_distance_mean": (
                    float(np.mean([
                        row["closest_observed_goal_distance"] for row in timeouts
                    ])) if timeouts else None
                ),
                "failures_with_early_approach_direction_reversal": sum(
                    row["early_approach_direction_reversals"] > 0
                    for row in records
                ),
                "episodes": records,
            }
    finally:
        environment.close()
    return summaries, examples


def plot_examples(examples: dict, task: dict) -> None:
    if set(examples) != {"collision", "timeout"}:
        raise RuntimeError("need both failure types in the seed-42 first draw")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    environment = run_environment(task)
    try:
        for axis, failure_type in zip(axes, ("collision", "timeout"), strict=True):
            scene_seed, result = examples[failure_type]
            initial = result.observations[0]
            axis.add_patch(Circle(initial[4:6], float(initial[6]), color="gray", alpha=0.3))
            for route, color in ((-1, "tab:blue"), (1, "tab:orange")):
                expert = TwoRouteExpert(route, dt=environment.dt,
                                        speed=float(task["expert_speed"]))
                reference = run_episode(environment, expert, seed=scene_seed)
                positions = reference.observations[:, :2]
                axis.plot(positions[:, 0], positions[:, 1], "--", color=color,
                          alpha=0.55, label=f"expert {route:+d}")
            positions = result.observations[:, :2]
            axis.plot(positions[:, 0], positions[:, 1], color="black", linewidth=2,
                      label="sampled policy")
            boundaries = positions[::int(task.get("execution_horizon", 4))]
            axis.scatter(boundaries[:, 0], boundaries[:, 1], s=13, color="black")
            axis.scatter([initial[0], initial[2]], [initial[1], initial[3]],
                         color=["green", "red"], zorder=3)
            axis.set(title=f"{failure_type}: scene {scene_seed}", xlabel="x (m)",
                     ylabel="y (m)",
                     xlim=(-1.45, max(1.45, float(positions[:, 0].max()) + 0.1)),
                     ylim=(-1.4, 1.4))
            axis.set_aspect("equal")
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / "dp_failure_examples.png", dpi=150)
    finally:
        environment.close()
        plt.close(figure)


def main() -> None:
    task = read_json(ROOT / "configs/053_diffusion_two_route_foundation.json")
    config = read_json(ROOT / "configs/053_diffusion_two_route_dp.json")
    foundation = read_json(OUTPUT_DIR / "results.json")
    source_hash = hashlib.sha256((OUTPUT_DIR / "dataset.npz").read_bytes()).hexdigest()
    if source_hash != foundation["dataset_sha256"]:
        raise RuntimeError("source dataset changed since expert audit")
    torch.set_num_threads(1)
    summaries, examples = replay_failures(task, config, source_hash)
    plot_examples(examples, {**task, "execution_horizon": config["execution_horizon"]})
    report = {
        "status": "failure_replay_not_causal_attribution",
        "source_dataset_sha256": source_hash,
        "replay_summaries": summaries,
    }
    with (OUTPUT_DIR / "dp_failure_analysis.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        seed: {key: value for key, value in summary.items() if key != "episodes"}
        for seed, summary in summaries.items()
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
