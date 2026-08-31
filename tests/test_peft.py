"""Unit tests for the readable LoRA implementation."""

import pytest
import torch
from torch import nn
from torch.optim import SGD

from robot_learning.peft import (
    LoRALinear,
    clone_linear_as_lora,
    inject_lora_into_linear_layers,
    merge_all_lora_layers,
)
from robot_learning.vision_language import VisionLanguageDualEncoder


def test_lora_initialization_preserves_base_output() -> None:
    torch.manual_seed(1)
    base = nn.Linear(5, 3)
    lora = clone_linear_as_lora(base, rank=2, alpha=4.0)
    inputs = torch.randn(7, 5)

    torch.testing.assert_close(lora(inputs), base(inputs))


def test_only_low_rank_parameters_are_trainable() -> None:
    lora = LoRALinear(nn.Linear(6, 4), rank=2, alpha=2.0)
    trainable_names = {
        name
        for name, parameter in lora.named_parameters()
        if parameter.requires_grad
    }

    assert trainable_names == {"lora_a.weight", "lora_b.weight"}
    assert lora.base_layer.weight.requires_grad is False
    assert lora.base_layer.bias.requires_grad is False


def test_trainable_parameter_count_matches_low_rank_formula() -> None:
    lora = LoRALinear(nn.Linear(12, 8), rank=3, alpha=6.0)

    assert lora.trainable_parameter_count == 3 * (12 + 8)


def test_first_backward_updates_b_before_a_receives_signal() -> None:
    torch.manual_seed(2)
    lora = LoRALinear(nn.Linear(5, 3), rank=2, alpha=2.0)
    inputs = torch.randn(8, 5)

    lora(inputs).square().mean().backward()

    assert torch.count_nonzero(lora.lora_b.weight.grad).item() > 0
    assert torch.count_nonzero(lora.lora_a.weight.grad).item() == 0
    assert lora.base_layer.weight.grad is None


def test_both_low_rank_matrices_train_after_b_leaves_zero() -> None:
    torch.manual_seed(3)
    lora = LoRALinear(nn.Linear(5, 3), rank=2, alpha=2.0)
    optimizer = SGD(lora.parameters(), lr=0.1)
    inputs = torch.randn(8, 5)

    optimizer.zero_grad(set_to_none=True)
    lora(inputs).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    lora(inputs).square().mean().backward()

    assert torch.count_nonzero(lora.lora_a.weight.grad).item() > 0
    assert torch.count_nonzero(lora.lora_b.weight.grad).item() > 0


def test_training_changes_output_without_changing_base_weight() -> None:
    torch.manual_seed(4)
    lora = LoRALinear(nn.Linear(4, 2), rank=2, alpha=2.0)
    optimizer = SGD(lora.parameters(), lr=0.1)
    inputs = torch.randn(6, 4)
    original_output = lora(inputs).detach().clone()
    original_base_weight = lora.base_layer.weight.detach().clone()

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        lora(inputs).square().mean().backward()
        optimizer.step()

    assert not torch.equal(original_output, lora(inputs))
    assert torch.equal(original_base_weight, lora.base_layer.weight)


def test_merged_linear_matches_trained_lora_in_evaluation_mode() -> None:
    torch.manual_seed(5)
    lora = LoRALinear(
        nn.Linear(4, 3),
        rank=2,
        alpha=4.0,
        dropout_probability=0.2,
    )
    with torch.no_grad():
        lora.lora_b.weight.normal_(mean=0.0, std=0.1)
    inputs = torch.randn(6, 4)
    lora.eval()

    merged = lora.to_merged_linear()

    torch.testing.assert_close(lora(inputs), merged(inputs))


@pytest.mark.parametrize(
    ("rank", "alpha", "dropout_probability"),
    ((0, 1.0, 0.0), (2, 0.0, 0.0), (2, 1.0, 1.0)),
)
def test_lora_rejects_invalid_configuration(
    rank: int,
    alpha: float,
    dropout_probability: float,
) -> None:
    with pytest.raises(ValueError):
        LoRALinear(
            nn.Linear(4, 3),
            rank=rank,
            alpha=alpha,
            dropout_probability=dropout_probability,
        )


def test_injection_preserves_initial_model_output() -> None:
    torch.manual_seed(10)
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    inputs = torch.randn(5, 4)
    original_output = model(inputs).detach().clone()

    report = inject_lora_into_linear_layers(
        model,
        target_module_names=("0", "2"),
        rank=2,
        alpha=4.0,
    )

    torch.testing.assert_close(model(inputs), original_output)
    assert report.replaced_module_names == ("0", "2")
    assert isinstance(model[0], LoRALinear)
    assert isinstance(model[2], LoRALinear)


def test_injection_freezes_unselected_parameters_and_counts_trainable_lora() -> None:
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))

    report = inject_lora_into_linear_layers(
        model,
        target_module_names=("2",),
        rank=2,
        alpha=2.0,
    )
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable_names == {"2.lora_a.weight", "2.lora_b.weight"}
    assert report.trainable_parameter_count == 2 * (6 + 2)
    assert 0.0 < report.trainable_fraction < 1.0


def test_injection_rejects_all_invalid_targets_before_modifying_model() -> None:
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))

    with pytest.raises(ValueError, match="unknown target"):
        inject_lora_into_linear_layers(
            model,
            target_module_names=("0", "missing"),
            rank=2,
            alpha=2.0,
        )

    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[2], nn.Linear)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_vlm_text_mlp_injection_only_trains_selected_lora_layers() -> None:
    torch.manual_seed(11)
    model = VisionLanguageDualEncoder(
        vocabulary_size=10,
        maximum_token_count=3,
        embedding_dimension=16,
        text_head_count=4,
        text_mlp_hidden_dimension=32,
        text_layer_count=1,
    )
    targets = (
        "text_encoder.transformer.layers.0.linear1",
        "text_encoder.transformer.layers.0.linear2",
    )

    report = inject_lora_into_linear_layers(
        model,
        targets,
        rank=2,
        alpha=4.0,
    )
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert report.replaced_module_names == targets
    assert trainable_names == {
        f"{targets[0]}.lora_a.weight",
        f"{targets[0]}.lora_b.weight",
        f"{targets[1]}.lora_a.weight",
        f"{targets[1]}.lora_b.weight",
    }


def test_injected_vlm_runs_forward_and_preserves_initial_embeddings() -> None:
    torch.manual_seed(13)
    model = VisionLanguageDualEncoder(
        vocabulary_size=10,
        maximum_token_count=3,
        embedding_dimension=16,
        text_head_count=4,
        text_mlp_hidden_dimension=32,
        text_layer_count=1,
    )
    images = torch.rand(4, 3, 32, 32)
    token_ids = torch.tensor(
        [[2, 3, 4], [2, 5, 6], [2, 7, 4], [2, 8, 6]],
        dtype=torch.int64,
    )
    model.eval()
    original = model(images, token_ids)

    inject_lora_into_linear_layers(
        model,
        (
            "text_encoder.transformer.layers.0.linear1",
            "text_encoder.transformer.layers.0.linear2",
        ),
        rank=2,
        alpha=4.0,
    )
    model.eval()
    injected = model(images, token_ids)

    torch.testing.assert_close(
        injected.image_embeddings,
        original.image_embeddings,
    )
    torch.testing.assert_close(
        injected.text_embeddings,
        original.text_embeddings,
    )


def test_vlm_evaluation_uses_nonzero_lora_and_matches_merged_model() -> None:
    torch.manual_seed(14)
    model = VisionLanguageDualEncoder(
        vocabulary_size=10,
        maximum_token_count=3,
        embedding_dimension=16,
        text_head_count=4,
        text_mlp_hidden_dimension=32,
        text_layer_count=1,
    )
    token_ids = torch.tensor(
        [[2, 3, 4], [2, 5, 6], [2, 7, 4], [2, 8, 6]],
        dtype=torch.int64,
    )
    model.eval()
    original = model.text_encoder(token_ids).detach().clone()
    targets = (
        "text_encoder.transformer.layers.0.linear1",
        "text_encoder.transformer.layers.0.linear2",
    )
    inject_lora_into_linear_layers(
        model,
        targets,
        rank=2,
        alpha=4.0,
    )
    with torch.no_grad():
        model.text_encoder.transformer.layers[0].linear1.lora_b.weight.normal_(
            std=0.2
        )
        model.text_encoder.transformer.layers[0].linear2.lora_b.weight.normal_(
            std=0.2
        )
    model.eval()
    adapted = model.text_encoder(token_ids).detach().clone()

    merge_all_lora_layers(model)
    merged = model.text_encoder(token_ids).detach().clone()

    assert not torch.allclose(adapted, original)
    torch.testing.assert_close(merged, adapted)


def test_merge_all_lora_layers_preserves_evaluation_output() -> None:
    torch.manual_seed(12)
    model = nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Linear(6, 2))
    inject_lora_into_linear_layers(
        model,
        target_module_names=("0", "2"),
        rank=2,
        alpha=4.0,
    )
    with torch.no_grad():
        model[0].lora_b.weight.normal_(std=0.1)
        model[2].lora_b.weight.normal_(std=0.1)
    inputs = torch.randn(5, 4)
    model.eval()
    expected = model(inputs).detach().clone()

    merged_names = merge_all_lora_layers(model)

    assert merged_names == ("0", "2")
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    assert not any(parameter.requires_grad for parameter in model.parameters())
    torch.testing.assert_close(model(inputs), expected)
