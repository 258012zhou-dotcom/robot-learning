"""Audit four-action chunks on an existing simulated trajectory dataset."""

import json
from pathlib import Path

import numpy as np

from robot_learning.action_chunk_dataset import build_action_chunks
from robot_learning.trajectory_dataset import load_transition_dataset


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/044_action_chunk_alignment.json"
OUTPUT_DIR = ROOT / "outputs/044_action_chunk_alignment"


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    dataset = load_transition_dataset(ROOT / config["source_dataset"])
    horizon = int(config["action_horizon"])
    chunks = build_action_chunks(dataset, horizon=horizon)
    if chunks.action_chunks.shape[0] != dataset.transition_count:
        raise RuntimeError("some end-of-Episode transitions were discarded")

    valid_counts = np.count_nonzero(chunks.valid_mask, axis=1)
    for source_row in range(dataset.transition_count):
        action_rows = chunks.action_row_indices[source_row, chunks.valid_mask[source_row]]
        if not np.all(dataset.episode_ids[action_rows] == dataset.episode_ids[source_row]):
            raise RuntimeError("an action chunk crossed an Episode boundary")
        if not np.all(dataset.split_ids[action_rows] == dataset.split_ids[source_row]):
            raise RuntimeError("an action chunk crossed a data split")
        expected_steps = dataset.step_ids[source_row] + np.arange(action_rows.size)
        if not np.array_equal(dataset.step_ids[action_rows], expected_steps):
            raise RuntimeError("an action chunk skipped a step")

    final_rows = np.flatnonzero(dataset.terminated | dataset.truncated)
    first_end = int(final_rows[0])
    next_start = first_end + 1
    if next_start >= dataset.transition_count:
        raise RuntimeError("the source has no second Episode for a boundary check")
    preview_rows = [0, first_end, next_start]
    report = {
        "experiment_name": config["experiment_name"],
        "source_dataset": config["source_dataset"],
        "source_type": "simulation, not real-robot data",
        "action_horizon": horizon,
        "episode_count": dataset.episode_count,
        "transition_count": dataset.transition_count,
        "chunk_count": int(chunks.action_chunks.shape[0]),
        "full_chunk_count": int(np.count_nonzero(valid_counts == horizon)),
        "padded_chunk_count": int(np.count_nonzero(valid_counts < horizon)),
        "padding_slots": int(np.sum(horizon - valid_counts)),
        "examples": [{
            "source_row": row,
            "episode_id": int(chunks.episode_ids[row]),
            "step_id": int(chunks.step_ids[row]),
            "split_id": int(chunks.split_ids[row]),
            "action_row_indices": chunks.action_row_indices[row].tolist(),
            "valid_mask": chunks.valid_mask[row].tolist(),
            "action_chunk": chunks.action_chunks[row].tolist(),
        } for row in preview_rows],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
