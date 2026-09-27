"""Hold an observation fixed and intervene on each ACT-style latent coordinate."""

import json
from pathlib import Path

import numpy as np
import torch

from robot_learning.act_lowdim import ACTLowDim
from robot_learning.sequence_behavior_cloning import prepare_sequence_data
from robot_learning.trajectory_dataset import load_transition_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "047_act_lowdim_mechanics"


def main() -> None:
    torch.set_num_threads(1)
    with (PROJECT_ROOT / "configs" / "047_act_lowdim_mechanics.json").open(
        encoding="utf-8"
    ) as file:
        config = json.load(file)
    with (PROJECT_ROOT / str(config["dataset_manifest_path"])).open(
        encoding="utf-8"
    ) as file:
        manifest = json.load(file)
    artifact = torch.load(
        OUTPUT_DIR / "best_model.pt", map_location="cpu", weights_only=True
    )
    if artifact["source_dataset_sha256"] != manifest["dataset_content_sha256"]:
        raise ValueError("dataset and checkpoint identity do not match")
    model = ACTLowDim(**artifact["model_config"])
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    model.eval()
    dataset = load_transition_dataset(PROJECT_ROOT / str(config["dataset_path"]))
    prepared = prepare_sequence_data(
        dataset,
        horizon=model.horizon,
        expert_policy_id=int(config["expert_policy_id"]),
    )
    np.testing.assert_allclose(
        prepared.normalization.mean, artifact["observation_mean"].numpy()
    )
    np.testing.assert_allclose(
        prepared.normalization.scale, artifact["observation_scale"].numpy()
    )
    # Exactly one held-out observation stays fixed throughout the intervention.
    observation = torch.from_numpy(prepared.test.observations[:1])
    zero = torch.zeros((1, model.latent_size))
    with torch.inference_mode():
        baseline = model(observation).actions
        torch.testing.assert_close(
            baseline, model.predict_with_latent(observation, zero)
        )
        per_axis = []
        for axis in range(model.latent_size):
            plus = zero.clone()
            minus = zero.clone()
            plus[0, axis] = 1.0
            minus[0, axis] = -1.0
            positive = model.predict_with_latent(observation, plus)
            negative = model.predict_with_latent(observation, minus)
            per_axis.append(
                {
                    "axis": axis,
                    "max_abs_delta_plus_one": float(
                        (positive - baseline).abs().max().item()
                    ),
                    "max_abs_delta_minus_one": float(
                        (negative - baseline).abs().max().item()
                    ),
                }
            )
        most_sensitive_axis = max(
            per_axis,
            key=lambda row: max(
                row["max_abs_delta_plus_one"], row["max_abs_delta_minus_one"]
            ),
        )["axis"]
        trajectories = []
        for value in (-2.0, -1.0, 0.0, 1.0, 2.0):
            latent = zero.clone()
            latent[0, most_sensitive_axis] = value
            actions = model.predict_with_latent(observation, latent)
            trajectories.append(
                {"latent_value": value, "actions": actions[0, :, 0].tolist()}
            )
    results = {
        "experiment_name": "047_act_lowdim_latent_intervention",
        "source_dataset_sha256": artifact["source_dataset_sha256"],
        "test_row_index": 0,
        "fixed_normalized_observation": observation[0].tolist(),
        "baseline_actions": baseline[0, :, 0].tolist(),
        "per_axis_unit_perturbation": per_axis,
        "most_sensitive_axis_on_this_observation": most_sensitive_axis,
        "trajectories": trajectories,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "latent_intervention.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
