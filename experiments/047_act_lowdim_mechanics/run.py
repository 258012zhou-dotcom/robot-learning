"""Train a low-dimensional ACT-style mechanism, without closed-loop claims."""

from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from robot_learning.act_lowdim import ACTLowDim, act_lowdim_loss
from robot_learning.sequence_behavior_cloning import SequenceSplit, prepare_sequence_data
from robot_learning.trajectory_dataset import load_transition_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "047_act_lowdim_mechanics.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "047_act_lowdim_mechanics"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return result


def make_loader(
    split: SequenceSplit, *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
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


def evaluate(
    model: ACTLowDim, loader: DataLoader, *, kl_weight: float, seed: int
) -> dict[str, float]:
    """Separate posterior reconstruction from deployable zero-latent predictions."""
    model.eval()
    posterior_l1_sum = 0.0
    prior_l1_sum = 0.0
    prior_first_l1_sum = 0.0
    valid_component_count = 0
    first_component_count = 0
    kl_sum = 0.0
    sample_count = 0
    # Validation posterior samples stay comparable across epochs; training RNG is untouched.
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(seed)
        for observations, actions, mask in loader:
            posterior = model(observations, actions, mask)
            prior = model(observations)
            _, _, kl = act_lowdim_loss(
                posterior, actions, mask, kl_weight=kl_weight
            )
            valid = mask.unsqueeze(-1)
            posterior_l1_sum += float(
                ((posterior.actions - actions).abs() * valid).sum().item()
            )
            prior_l1_sum += float(
                ((prior.actions - actions).abs() * valid).sum().item()
            )
            prior_first_l1_sum += float(
                (prior.actions[:, 0] - actions[:, 0]).abs().sum().item()
            )
            valid_component_count += int(mask.sum()) * model.action_size
            first_component_count += int(observations.shape[0]) * model.action_size
            kl_sum += float(kl.item()) * int(observations.shape[0])
            sample_count += int(observations.shape[0])
    return {
        "posterior_chunk_l1": posterior_l1_sum / valid_component_count,
        "prior_chunk_l1": prior_l1_sum / valid_component_count,
        "prior_first_action_l1": prior_first_l1_sum / first_component_count,
        "mean_kl": kl_sum / sample_count,
    }


def main() -> None:
    config = read_json(CONFIG_PATH)
    if config["device"] != "cpu":
        raise ValueError("this teaching experiment uses CPU only")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    dataset = load_transition_dataset(PROJECT_ROOT / str(config["dataset_path"]))
    manifest = read_json(PROJECT_ROOT / str(config["dataset_manifest_path"]))
    source_hash = str(manifest["dataset_content_sha256"])
    prepared = prepare_sequence_data(
        dataset,
        horizon=int(config["horizon"]),
        expert_policy_id=int(config["expert_policy_id"]),
    )
    loaders = {
        name: make_loader(
            getattr(prepared, name),
            batch_size=int(config["batch_size"]),
            shuffle=name == "train",
            seed=seed,
        )
        for name in ("train", "validation", "test")
    }
    model_config = {
        "observation_size": int(prepared.train.observations.shape[1]),
        "action_size": int(prepared.train.action_chunks.shape[2]),
        "horizon": int(config["horizon"]),
        "hidden_size": int(config["hidden_size"]),
        "latent_size": int(config["latent_size"]),
        "attention_heads": int(config["attention_heads"]),
    }
    model = ACTLowDim(**model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    kl_weight = float(config["kl_weight"])
    initial_validation = evaluate(
        model, loaders["validation"], kl_weight=kl_weight, seed=seed + 1
    )
    best_prior_l1 = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        for observations, actions, mask in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            output = model(observations, actions, mask)
            total, _, _ = act_lowdim_loss(
                output, actions, mask, kl_weight=kl_weight
            )
            total.backward()
            optimizer.step()
        validation = evaluate(
            model, loaders["validation"], kl_weight=kl_weight, seed=seed + 1
        )
        history.append({"epoch": epoch, **validation})
        if validation["prior_chunk_l1"] < best_prior_l1:
            best_prior_l1 = validation["prior_chunk_l1"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "best_model.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "observation_mean": torch.from_numpy(prepared.normalization.mean.copy()),
            "observation_scale": torch.from_numpy(prepared.normalization.scale.copy()),
            "source_dataset_sha256": source_hash,
            "seed": seed,
        },
        checkpoint_path,
    )
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if artifact["source_dataset_sha256"] != source_hash:
        raise ValueError("checkpoint dataset identity mismatch")
    reloaded = ACTLowDim(**artifact["model_config"])
    reloaded.load_state_dict(artifact["model_state_dict"], strict=True)
    reloaded.eval()
    example = torch.from_numpy(prepared.test.observations[:2])
    with torch.inference_mode():
        torch.testing.assert_close(model(example).actions, reloaded(example).actions)

    test = evaluate(
        reloaded, loaders["test"], kl_weight=kl_weight, seed=seed + 2
    )
    results = {
        "experiment_name": config["experiment_name"],
        "status": "lowdim_act_mechanism_not_paper_reproduction",
        "source_dataset_sha256": source_hash,
        "seed": seed,
        "model_config": model_config,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "split_rows": {
            name: int(getattr(prepared, name).observations.shape[0])
            for name in ("train", "validation", "test")
        },
        "split_episodes": {
            name: int(np.unique(getattr(prepared, name).episode_ids).size)
            for name in ("train", "validation", "test")
        },
        "initial_validation": initial_validation,
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1],
        "test": test,
        "validation_by_epoch": history,
        "checkpoint_reload_matches": True,
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(json.dumps({key: value for key, value in results.items() if key != "validation_by_epoch"}, indent=2))


if __name__ == "__main__":
    main()
