"""Mechanism-only probe: two opposite action chunks share one observation."""

import json
from pathlib import Path
import random

import numpy as np
import torch

from robot_learning.act_bimodal_probe import make_opposite_action_pair
from robot_learning.act_lowdim import ACTLowDim, act_lowdim_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "048_bimodal_act_probe.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "048_bimodal_act_probe"


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    observations, targets, mask = make_opposite_action_pair(
        horizon=int(config["horizon"]),
        magnitude=float(config["action_magnitude"]),
    )
    model = ACTLowDim(
        observation_size=observations.shape[1],
        action_size=targets.shape[2],
        horizon=targets.shape[1],
        hidden_size=int(config["hidden_size"]),
        latent_size=int(config["latent_size"]),
        attention_heads=int(config["attention_heads"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    losses = []
    for step in range(1, int(config["training_steps"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        posterior = model(observations, targets, mask)
        total, reconstruction, kl = act_lowdim_loss(
            posterior, targets, mask, kl_weight=float(config["kl_weight"])
        )
        total.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            losses.append(
                {
                    "step": step,
                    "sampled_posterior_l1": float(reconstruction.item()),
                    "kl": float(kl.item()),
                }
            )

    model.eval()
    with torch.inference_mode():
        posterior = model(observations, targets, mask)
        assert posterior.posterior_mean is not None
        assert posterior.posterior_logvar is not None
        decoded_at_means = model.predict_with_latent(
            observations, posterior.posterior_mean
        )
        default = model(observations).actions
        torch.testing.assert_close(default[0], default[1])
        reconstruction_l1 = (
            (decoded_at_means - targets).abs().mean(dim=(1, 2)).tolist()
        )
        default_l1 = ((default - targets).abs().mean(dim=(1, 2)).tolist())
        _, _, final_kl = act_lowdim_loss(
            posterior, targets, mask, kl_weight=float(config["kl_weight"])
        )

    results = {
        "experiment_name": config["experiment_name"],
        "status": "synthetic_two_example_mechanism_probe_not_closed_loop",
        "seed": seed,
        "training_steps": int(config["training_steps"]),
        "targets": targets[:, :, 0].tolist(),
        "posterior_means": posterior.posterior_mean.tolist(),
        "posterior_mean_reconstruction": decoded_at_means[:, :, 0].tolist(),
        "posterior_mean_reconstruction_l1_by_mode": reconstruction_l1,
        "default_zero_latent_chunk": default[0, :, 0].tolist(),
        "default_zero_latent_l1_by_mode": default_l1,
        "mean_kl": float(final_kl.item()),
        "training_snapshots": losses,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(json.dumps({key: value for key, value in results.items() if key != "training_snapshots"}, indent=2))


if __name__ == "__main__":
    main()
