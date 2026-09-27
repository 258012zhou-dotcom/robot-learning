"""Print the exact predictions and weights used at one absolute time."""

import json
from pathlib import Path

import numpy as np

from robot_learning.temporal_ensemble import ensemble_action_at


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "049_temporal_ensemble_alignment.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "049_temporal_ensemble_alignment"


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    chunks = {
        int(step): np.asarray(actions, dtype=np.float32)
        for step, actions in config["chunks_by_issued_step"].items()
    }
    result = ensemble_action_at(
        chunks,
        target_step=int(config["target_step"]),
        decay=float(config["decay"]),
    )
    rows = [
        {
            "issued_step": issued_step,
            "chunk_offset": offset,
            "absolute_step": issued_step + offset,
            "aligned_action": action.tolist(),
            "normalized_weight": weight,
        }
        for issued_step, offset, action, weight in zip(
            result.issued_steps,
            result.chunk_offsets,
            result.aligned_predictions,
            result.normalized_weights,
            strict=True,
        )
    ]
    output = {
        "experiment_name": config["experiment_name"],
        "target_step": int(config["target_step"]),
        "decay": float(config["decay"]),
        "contributors": rows,
        "ensemble_action": result.action.tolist(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
