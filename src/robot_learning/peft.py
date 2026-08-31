"""Readable parameter-efficient fine-tuning components."""

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """Add a trainable low-rank update to a frozen linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be torch.nn.Linear")
        if type(rank) is not int or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not isinstance(alpha, Real) or isinstance(alpha, bool):
            raise ValueError("alpha must be a positive finite number")
        if not math.isfinite(float(alpha)) or alpha <= 0.0:
            raise ValueError("alpha must be a positive finite number")
        if (
            not isinstance(dropout_probability, Real)
            or isinstance(dropout_probability, bool)
            or not math.isfinite(float(dropout_probability))
            or not 0.0 <= dropout_probability < 1.0
        ):
            raise ValueError("dropout_probability must be in [0, 1)")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout_probability))
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        factory_options = {
            "device": self.base_layer.weight.device,
            "dtype": self.base_layer.weight.dtype,
        }
        self.lora_a = nn.Linear(
            self.base_layer.in_features,
            rank,
            bias=False,
            **factory_options,
        )
        self.lora_b = nn.Linear(
            rank,
            self.base_layer.out_features,
            bias=False,
            **factory_options,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_b.weight)

    @property
    def in_features(self) -> int:
        """Expose the wrapped input dimension."""
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        """Expose the wrapped output dimension."""
        return self.base_layer.out_features

    @property
    def weight(self) -> Tensor:
        """Expose the effective weight for framework fused evaluation paths."""
        return self.merged_weight()

    @property
    def bias(self) -> Tensor | None:
        """Expose the frozen base bias for nn.Linear-compatible access."""
        return self.base_layer.bias

    @property
    def trainable_parameter_count(self) -> int:
        """Return the number of parameters added by the low-rank update."""
        return sum(
            parameter.numel()
            for parameter in (self.lora_a.weight, self.lora_b.weight)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return frozen base output plus the scaled low-rank update."""
        base_output = self.base_layer(inputs)
        low_rank_update = self.lora_b(self.lora_a(self.dropout(inputs)))
        return base_output + self.scaling * low_rank_update

    def merged_weight(self) -> Tensor:
        """Return W + scaling * B @ A without modifying the base layer."""
        update = self.lora_b.weight @ self.lora_a.weight
        return self.base_layer.weight + self.scaling * update

    def to_merged_linear(self) -> nn.Linear:
        """Create a frozen ordinary linear layer for merged inference."""
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.base_layer.bias is not None,
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self.merged_weight())
            if self.base_layer.bias is not None:
                merged.bias.copy_(self.base_layer.bias)
        for parameter in merged.parameters():
            parameter.requires_grad = False
        return merged


def clone_linear_as_lora(
    linear: nn.Linear,
    rank: int,
    alpha: float,
    dropout_probability: float = 0.0,
) -> LoRALinear:
    """Deep-copy a linear layer before wrapping it with LoRA."""
    return LoRALinear(
        deepcopy(linear),
        rank,
        alpha,
        dropout_probability,
    )


@dataclass(frozen=True)
class LoRAInjectionReport:
    """Names and parameter counts after selected LoRA injection."""

    replaced_module_names: tuple[str, ...]
    total_parameter_count: int
    trainable_parameter_count: int

    @property
    def trainable_fraction(self) -> float:
        """Return trainable parameters divided by all stored parameters."""
        return self.trainable_parameter_count / self.total_parameter_count


def _parent_and_child_module(
    model: nn.Module,
    module_name: str,
) -> tuple[nn.Module, str]:
    """Resolve the parent module and final attribute of a dotted path."""
    if not module_name or module_name.startswith(".") or module_name.endswith("."):
        raise ValueError("target module names must be non-empty dotted paths")
    parent_name, separator, child_name = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if separator else model
    return parent, child_name


def inject_lora_into_linear_layers(
    model: nn.Module,
    target_module_names: list[str] | tuple[str, ...],
    rank: int,
    alpha: float,
    dropout_probability: float = 0.0,
) -> LoRAInjectionReport:
    """Freeze a model and replace selected linear modules with LoRA wrappers."""
    if not target_module_names:
        raise ValueError("target_module_names must not be empty")
    if len(set(target_module_names)) != len(target_module_names):
        raise ValueError("target_module_names must not contain duplicates")

    resolved_targets: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for module_name in target_module_names:
        try:
            parent, child_name = _parent_and_child_module(model, module_name)
            target = getattr(parent, child_name)
        except (AttributeError, KeyError) as error:
            raise ValueError(f"unknown target module: {module_name}") from error
        if not isinstance(target, nn.Linear):
            raise TypeError(f"target module is not nn.Linear: {module_name}")
        resolved_targets.append((module_name, parent, child_name, target))

    for parameter in model.parameters():
        parameter.requires_grad = False
    for _, parent, child_name, target in resolved_targets:
        setattr(
            parent,
            child_name,
            LoRALinear(target, rank, alpha, dropout_probability),
        )

    total_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return LoRAInjectionReport(
        replaced_module_names=tuple(target_module_names),
        total_parameter_count=total_count,
        trainable_parameter_count=trainable_count,
    )


def merge_all_lora_layers(model: nn.Module) -> tuple[str, ...]:
    """Replace every LoRA wrapper with a frozen merged linear layer in place."""
    lora_module_names = tuple(
        name
        for name, module in model.named_modules()
        if name and isinstance(module, LoRALinear)
    )
    for module_name in lora_module_names:
        parent, child_name = _parent_and_child_module(model, module_name)
        lora_module = getattr(parent, child_name)
        setattr(parent, child_name, lora_module.to_merged_linear())
    return lora_module_names
