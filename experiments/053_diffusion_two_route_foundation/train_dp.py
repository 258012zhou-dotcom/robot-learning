"""Train/evaluate a low-dimensional diffusion policy on the frozen 053 task."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from robot_learning.gymnasium_rollout import run_episode
from robot_learning.lowdim_action_diffusion import (
    DiffusionChunkPolicy,
    LowDimActionDiffusion,
    load_diffusion_checkpoint,
    masked_noise_mse,
    save_diffusion_checkpoint,
)
from robot_learning.sequence_behavior_cloning import SequenceSplit, prepare_sequence_data
from robot_learning.trajectory_dataset import load_transition_dataset
from robot_learning.two_route_navigation import TwoRouteNavigationEnv


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/053_diffusion_two_route_foundation"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def make_loader(split: SequenceSplit, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(split.observations),
            torch.from_numpy(split.action_chunks),
            torch.from_numpy(split.valid_mask),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def denoising_mse(
    model: LowDimActionDiffusion, data: DataLoader, *, noise_seed: int
) -> float:
    """Use a fixed noise/timestep stream so checkpoints have comparable validation loss."""
    model.eval()
    generator = torch.Generator().manual_seed(noise_seed)
    total_error = 0.0
    valid_components = 0
    with torch.inference_mode():
        for observations, actions, mask in data:
            timesteps = torch.randint(
                model.diffusion_steps, (len(observations),), generator=generator
            )
            noise = torch.randn(actions.shape, generator=generator)
            noisy = model.add_noise(actions, timesteps, noise) * mask.unsqueeze(-1)
            predicted = model(observations, noisy, timesteps)
            component_count = int(mask.sum()) * model.action_size
            total_error += float(masked_noise_mse(predicted, noise, mask)) * component_count
            valid_components += component_count
    return total_error / valid_components


def train(
    model: LowDimActionDiffusion,
    training: DataLoader,
    validation: DataLoader,
    config: dict,
) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        for observations, actions, mask in training:
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(observations, actions, mask)
            loss.backward()
            optimizer.step()
        validation_loss = denoising_mse(model, validation, noise_seed=2026)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch % 20 == 0:
            print(f"epoch {epoch}: validation noise MSE={validation_loss:.4f}", flush=True)
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_validation_noise_mse": best_loss}


def initial_chunk_audit(
    model: LowDimActionDiffusion, normalization, dataset, config: dict, task: dict
) -> dict:
    """On validation scene only: check if samples choose coherent route prefixes."""
    scene_seed = int(task["scene_splits"]["validation"]["seed_start"])
    rows = np.flatnonzero((dataset.environment_seeds == scene_seed) &
                          (dataset.step_ids == 0))
    if len(rows) != 2:
        raise RuntimeError("validation scene must contain two first actions")
    first = dataset.observations[rows[0]]
    if not np.array_equal(first, dataset.observations[rows[1]]):
        raise RuntimeError("route initial observations differ")
    count = int(config["initial_chunk_sample_count"])
    observations = normalization.transform(np.repeat(first[None, :], count, axis=0))
    chunks = model.sample(
        torch.from_numpy(observations), generator=torch.Generator().manual_seed(404)
    ).cpu().numpy()
    execution = int(config["execution_horizon"])
    mean_vertical = chunks[:, :execution, 1].mean(axis=1)
    upper = int(np.sum(mean_vertical > 0.1))
    lower = int(np.sum(mean_vertical < -0.1))
    neutral = count - upper - lower
    vertical = chunks[:, :execution, 1]
    switches = int(np.sum((vertical > 0.1).any(axis=1) & (vertical < -0.1).any(axis=1)))
    return {
        "validation_scene_seed": scene_seed,
        "sample_count": count,
        "expert_first_actions_by_route": {
            str(int(dataset.policy_ids[row])): dataset.actions[row].tolist()
            for row in rows
        },
        "first_action_vertical_mean": float(chunks[:, 0, 1].mean()),
        "first_action_vertical_std": float(chunks[:, 0, 1].std()),
        "first_four_mean_vertical_upper_count": upper,
        "first_four_mean_vertical_lower_count": lower,
        "first_four_mean_vertical_neutral_count": neutral,
        "first_four_with_strong_sign_switch_count": switches,
    }


def evaluate_closed_loop(
    policy: DiffusionChunkPolicy, seeds: list[int], task: dict,
    *, draws: int, training_seed: int
) -> dict:
    environment = TwoRouteNavigationEnv(
        dt=float(task["dt"]),
        max_episode_steps=int(task["max_episode_steps"]),
        goal_tolerance=float(task["goal_tolerance"]),
    )
    records = []
    try:
        for scene_seed in seeds:
            for draw in range(draws):
                sampling_seed = training_seed + scene_seed * 100 + draw
                policy.reset(seed=sampling_seed)
                result = run_episode(environment, policy, seed=scene_seed)
                start = result.observations[0]
                records.append({
                    "scene_seed": scene_seed,
                    "draw": draw,
                    "sampling_seed": sampling_seed,
                    "success": bool(result.is_success),
                    "collision": bool(result.terminated and not result.is_success),
                    "timeout": bool(result.truncated),
                    "steps": result.step_count,
                    "first_action": result.actions[0].tolist(),
                    "final_distance": float(np.linalg.norm(
                        result.observations[-1, :2] - start[2:4]
                    )),
                })
    finally:
        environment.close()
    first_draw = [row for row in records if row["draw"] == 0]
    return {
        "scene_count": len(seeds),
        "draws_per_scene": draws,
        "rollout_count": len(records),
        "success_count": sum(row["success"] for row in records),
        "collision_count": sum(row["collision"] for row in records),
        "timeout_count": sum(row["timeout"] for row in records),
        "first_draw_success_count": sum(row["success"] for row in first_draw),
        "first_draw_collision_count": sum(row["collision"] for row in first_draw),
        "episodes": records,
    }


def main(seed_override: int | None = None) -> None:
    config = read_json(ROOT / "configs/053_diffusion_two_route_dp.json")
    task = read_json(ROOT / "configs/053_diffusion_two_route_foundation.json")
    foundation = read_json(OUTPUT_DIR / "results.json")
    dataset_path = OUTPUT_DIR / "dataset.npz"
    source_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if source_hash != foundation["dataset_sha256"]:
        raise RuntimeError("dataset changed since expert audit")
    seed = int(config["seed"] if seed_override is None else seed_override)
    if seed < 0:
        raise ValueError("training seed must be non-negative")
    if seed not in config["repeat_train_seeds"]:
        raise ValueError("training seed is not in the fixed repeat_train_seeds list")
    run_config = {**config, "seed": seed}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    dataset = load_transition_dataset(dataset_path)
    prepared = prepare_sequence_data(
        dataset, horizon=int(config["prediction_horizon"]), expert_policy_ids=[0, 1]
    )
    loaders = {
        name: make_loader(
            getattr(prepared, name), batch_size=int(config["batch_size"]),
            shuffle=name == "train", seed=seed,
        )
        for name in ("train", "validation", "test")
    }
    model = LowDimActionDiffusion(
        7, 2, horizon=int(config["prediction_horizon"]),
        diffusion_steps=int(config["diffusion_steps"]),
        hidden_size=int(config["hidden_size"]),
    )
    initial_validation_mse = denoising_mse(model, loaders["validation"], noise_seed=2026)
    history = train(model, loaders["train"], loaders["validation"], config)
    artifact_stem = "dp" if seed == int(config["seed"]) else f"dp_seed_{seed}"
    checkpoint_path = OUTPUT_DIR / f"{artifact_stem}_best_model.pt"
    save_diffusion_checkpoint(
        checkpoint_path, model, prepared.normalization,
        source_dataset_sha256=source_hash, seed=seed,
    )
    model, normalization, artifact = load_diffusion_checkpoint(checkpoint_path)
    if artifact["source_dataset_sha256"] != source_hash:
        raise RuntimeError("checkpoint and dataset identity disagree")
    policy = DiffusionChunkPolicy(
        model, normalization, execution_horizon=int(config["execution_horizon"])
    )
    draws = int(config["evaluation_draws_per_scene"])
    if draws <= 0:
        raise ValueError("evaluation_draws_per_scene must be positive")
    scene_seeds = {
        name: list(range(
            int(task["scene_splits"][name]["seed_start"]),
            int(task["scene_splits"][name]["seed_start"])
            + int(task["scene_splits"][name]["scene_count"]),
        ))
        for name in ("validation", "test")
    }
    report = {
        "status": "lowdim_diffusion_one_seed_teaching_pilot_not_paper_reproduction",
        "source_dataset_sha256": source_hash,
        "config": run_config,
        "split_transition_counts": {
            name: len(getattr(prepared, name).observations) for name in loaders
        },
        "initial_validation_noise_mse": initial_validation_mse,
        "training": history,
        "validation_noise_mse": denoising_mse(
            model, loaders["validation"], noise_seed=2026
        ),
        "test_noise_mse": denoising_mse(model, loaders["test"], noise_seed=2027),
        "initial_chunk_audit": initial_chunk_audit(
            model, normalization, dataset, config, task
        ),
        "validation_closed_loop": evaluate_closed_loop(
            policy, scene_seeds["validation"], task, draws=draws, training_seed=seed
        ),
        "test_closed_loop": evaluate_closed_loop(
            policy, scene_seeds["test"], task, draws=draws, training_seed=seed
        ),
    }
    with (OUTPUT_DIR / f"{artifact_stem}_results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        "seed": seed,
        "best_epoch": history["best_epoch"],
        "test_noise_mse": report["test_noise_mse"],
        "initial_chunk_audit": report["initial_chunk_audit"],
        "validation_success": report["validation_closed_loop"]["success_count"],
        "test_success": report["test_closed_loop"]["success_count"],
        "test_first_draw_success": report["test_closed_loop"]["first_draw_success_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="one seed from config repeat_train_seeds")
    arguments = parser.parse_args()
    main(arguments.seed)
