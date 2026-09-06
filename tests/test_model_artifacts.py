"""Checkpoint round trips and real inference entry points, using temporary files."""

import importlib.util
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

from robot_learning.model_artifacts import (
    image_preprocessing,
    load_model_artifact,
    save_model_artifact,
)
from robot_learning.object_detection import SmallShapeDetector
from robot_learning.peft import LoRALinear, inject_lora_into_linear_layers
from robot_learning.semantic_segmentation import SmallSemanticSegmenter
from robot_learning.vision_language import SimpleVocabulary


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "shape_detector": "008_single_object_detection",
    "semantic_segmenter": "009_semantic_segmentation",
    "lora_dual_encoder": "015_lora_language_adaptation",
}


def load_runner(kind):
    path = ROOT / "experiments" / EXPERIMENTS[kind] / "run.py"
    spec = importlib.util.spec_from_file_location(f"runner_{kind}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_checkpoint(tmp_path, kind):
    path = tmp_path / f"{kind}.pt"
    if kind == "lora_dual_encoder":
        runner = load_runner(kind)
        vocabulary = SimpleVocabulary.from_texts(("a red square",) + runner.TARGET_DESCRIPTIONS)
        # Reverse ordinary token IDs: rebuilding a sorted vocabulary would be wrong.
        words = list(vocabulary.token_to_id)[2:][::-1]
        vocabulary = SimpleVocabulary({
            token: index for index, token in enumerate(["<PAD>", "<UNK>"] + words)
        })
        config = {
            "maximum_token_count": 4, "embedding_dimension": 8,
            "text_head_count": 2, "text_mlp_hidden_dimension": 16,
            "text_layer_count": 1, "lora_rank": 2, "lora_alpha": 3.0,
            "lora_dropout_probability": 0.2, "image_size": 32,
        }
        model = runner.create_model(config, vocabulary)
        inject_lora_into_linear_layers(model, runner.LORA_TARGETS, 2, 3.0, 0.2)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.lora_b.weight.normal_(std=0.1)
        runner.save_lora_checkpoint(
            path, model, config, vocabulary, ("a red square",),
            {"best_epoch": 2, "best_validation_loss": 0.3},
        )
    else:
        classes = ["square", "circle"] if kind == "shape_detector" else ["background", "square", "circle"]
        model_type = SmallShapeDetector if kind == "shape_detector" else SmallSemanticSegmenter
        model = model_type(len(classes))
        save_model_artifact(path, model, {
            "model_kind": kind, "architecture": {"num_classes": len(classes)},
            "class_names": classes, "preprocessing": image_preprocessing(32),
            "best_epoch": 2, "best_validation_loss": 0.3,
        })
    return path, model.eval()


@pytest.mark.parametrize("kind", EXPERIMENTS)
@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
def test_round_trip_preserves_every_weight_output_and_device(tmp_path, kind, device):
    path, original = make_checkpoint(tmp_path, kind)
    original.to(device)
    # A different random initialization must not affect restored predictions.
    torch.manual_seed(987)
    loaded, artifact = load_model_artifact(path, device, expected_kind=kind)
    assert not loaded.training
    assert original.state_dict().keys() == loaded.state_dict().keys()
    for name, value in original.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
        assert loaded.state_dict()[name].device == torch.device(device)
    images = torch.rand(2, 3, 32, 32, device=device)
    with torch.inference_mode():
        if kind == "lora_dual_encoder":
            assert artifact["vocabulary_tokens"][2:] == sorted(
                artifact["vocabulary_tokens"][2:], reverse=True
            )
            vocabulary = SimpleVocabulary({
                token: index for index, token in enumerate(artifact["vocabulary_tokens"])
            })
            tokens = vocabulary.encode_batch(["crimson box", "azure disk"], 4).to(device)
            before, after = original(images, tokens), loaded(images, tokens)
            pairs = [(before.image_embeddings, after.image_embeddings),
                     (before.text_embeddings, after.text_embeddings)]
            assert any("base_layer.weight" in key for key in artifact["state_dict"])
            assert any("image_encoder" in key for key in artifact["state_dict"])
            assert all(module.lora_b.weight.device == module.base_layer.weight.device
                       for module in loaded.modules() if isinstance(module, LoRALinear))
        elif kind == "shape_detector":
            pairs = zip(original(images), loaded(images))
        else:
            pairs = [(original(images), loaded(images))]
        for before, after in pairs:
            torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("kind", EXPERIMENTS)
@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_strict_loading_rejects_damaged_weights(tmp_path, kind, corruption):
    path, _ = make_checkpoint(tmp_path, kind)
    artifact = torch.load(path, weights_only=True)
    state = artifact["state_dict"]
    key = next(iter(state))
    if corruption == "missing":
        del state[key]
    elif corruption == "unexpected":
        state["unexpected.weight"] = torch.ones(1)
    else:
        state[key] = torch.zeros(1)
    torch.save(artifact, path)
    with pytest.raises(RuntimeError):
        load_model_artifact(path)


@pytest.mark.parametrize("kind", EXPERIMENTS)
def test_real_inference_entry_point(tmp_path, kind):
    path, _ = make_checkpoint(tmp_path, kind)
    image = tmp_path / "input.png"
    assert cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8))
    # Separate process with no training config override or training invocation.
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / EXPERIMENTS[kind] / "run.py"),
         "--checkpoint", str(path), "--image", str(image), "--device", "cpu"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
                          "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg"},
        capture_output=True, text=True, check=True, timeout=60,
    )
    output = json.loads(result.stdout)
    if kind == "shape_detector":
        assert output["class_name"] in ("square", "circle")
        assert len(output["box"]) == 4
        assert all(0 <= value <= 1 for value in output["box"])
    elif kind == "semantic_segmenter":
        mask = np.array(output["mask"])
        assert mask.shape == (32, 32)
        assert set(np.unique(mask)) <= {0, 1, 2}
    else:
        assert len(output["cosine_similarities"]) == 6
        assert output["prediction"] in output["descriptions"]


def test_rejects_legacy_adapter_and_wrong_experiment(tmp_path):
    legacy = tmp_path / "adapter.pt"
    torch.save({"lora_a.weight": torch.ones(2, 2)}, legacy)
    with pytest.raises(ValueError, match="legacy adapter"):
        load_model_artifact(legacy)
    path, _ = make_checkpoint(tmp_path, "shape_detector")
    with pytest.raises(ValueError, match="expected semantic_segmenter"):
        load_model_artifact(path, expected_kind="semantic_segmenter")


@pytest.mark.parametrize("kind", ["shape_detector", "semantic_segmenter"])
def test_training_entry_saves_best_not_last_epoch(tmp_path, monkeypatch, kind):
    runner = load_runner(kind)
    config = runner.load_config()
    config.update(sample_count=24, image_size=32, batch_size=8, epochs=2)
    monkeypatch.setattr(runner, "load_config", lambda: config)
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(runner, "select_torch_device", lambda: torch.device("cpu"))
    for name in vars(runner):
        if name.startswith("save_") and name != "save_model_artifact":
            monkeypatch.setattr(runner, name, lambda *args: None)
    module_name, evaluation_name, loss_name = (
        ("object_detection", "evaluate_detector", "total_loss")
        if kind == "shape_detector" else
        ("semantic_segmentation", "evaluate_segmenter", "loss")
    )
    training_module = importlib.import_module(f"robot_learning.{module_name}")
    real_evaluate = getattr(training_module, evaluation_name)
    validation_states = []

    def controlled_validation(model, *args, **kwargs):
        evaluation = real_evaluate(model, *args, **kwargs)
        validation_states.append({key: value.detach().clone()
                                  for key, value in model.state_dict().items()})
        # First epoch wins deliberately, even though training continues.
        return replace(evaluation, **{loss_name: float(len(validation_states))})

    monkeypatch.setattr(training_module, evaluation_name, controlled_validation)
    runner.main()
    loaded, artifact = load_model_artifact(tmp_path / "best_model.pt")
    assert artifact["best_epoch"] == 1
    assert artifact["best_validation_loss"] == 1.0
    assert artifact["class_names"] == list(runner.CLASS_NAMES)
    assert artifact["preprocessing"] == image_preprocessing(32)
    assert any(not torch.equal(value, validation_states[1][key])
               for key, value in validation_states[0].items())
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, validation_states[0][key]), key


def test_lora_training_entry_emits_complete_checkpoint(tmp_path, monkeypatch):
    runner = load_runner("lora_dual_encoder")
    config = runner.load_config()
    for name in config:
        if name.endswith("samples_per_concept"):
            config[name] = 1
    config.update(source_epochs=1, target_epochs=1, trial_count=1,
                  embedding_dimension=8, text_head_count=2,
                  text_mlp_hidden_dimension=16, lora_rank=2)
    monkeypatch.setattr(runner, "load_config", lambda: config)
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(runner, "select_torch_device", lambda: torch.device("cpu"))
    for name in ("save_accuracy_comparison", "save_parameter_comparison", "save_first_trial_curves"):
        monkeypatch.setattr(runner, name, lambda *args: None)
    runner.main()
    model, artifact = load_model_artifact(tmp_path / "trial_1_lora_model.pt")
    assert artifact["best_epoch"] == 1
    assert artifact["lora"]["rank"] == 2
    assert any(torch.count_nonzero(module.lora_b.weight) > 0
               for module in model.modules() if isinstance(module, LoRALinear))
    assert not (tmp_path / "trial_1_lora_adapter.pt").exists()
