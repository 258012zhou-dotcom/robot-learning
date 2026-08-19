"""用于小型学习动力学实验的数据生成与划分。"""

from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor


@dataclass(frozen=True)
class MotionDatasetSplits:
    """训练、验证和测试数据及其原始样本索引。"""

    train_inputs: Tensor
    train_targets: Tensor
    validation_inputs: Tensor
    validation_targets: Tensor
    test_inputs: Tensor
    test_targets: Tensor
    train_indices: Tensor
    validation_indices: Tensor
    test_indices: Tensor


@dataclass(frozen=True)
class StandardizationStats:
    """由训练集拟合得到的逐特征标准化参数。"""

    mean: Tensor
    standard_deviation: Tensor


def generate_motion_samples(
    sample_count: int,
    dt: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """生成二维匀速运动样本。

    输入列依次为 x、y、vx、vy，目标为经过 dt 后的二维位置。
    """
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if not isinstance(dt, Real) or isinstance(dt, bool) or dt <= 0:
        raise ValueError("dt must be a positive number")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    positions = torch.empty(sample_count, 2).uniform_(
        -10.0,
        10.0,
        generator=generator,
    )
    velocities = torch.empty(sample_count, 2).uniform_(
        -2.0,
        2.0,
        generator=generator,
    )

    inputs = torch.cat((positions, velocities), dim=1)
    targets = positions + velocities * float(dt)
    return inputs, targets


def split_motion_dataset(
    inputs: Tensor,
    targets: Tensor,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> MotionDatasetSplits:
    """用互不重叠的随机索引划分训练、验证和测试数据。"""
    if inputs.ndim != 2 or inputs.shape[1] != 4:
        raise ValueError("inputs must have shape (N, 4)")
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets must have shape (N, 2)")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must contain the same sample count")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    for value, name in (
        (train_fraction, "train_fraction"),
        (validation_fraction, "validation_fraction"),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not 0 < value < 1
        ):
            raise ValueError(f"{name} must be between 0 and 1")

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "train_fraction and validation_fraction must leave a test split"
        )

    sample_count = inputs.shape[0]
    train_count = int(sample_count * train_fraction)
    validation_count = int(sample_count * validation_fraction)
    test_count = sample_count - train_count - validation_count
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("each dataset split must contain at least one sample")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    shuffled_indices = torch.randperm(sample_count, generator=generator)
    train_end = train_count
    validation_end = train_count + validation_count

    train_indices = shuffled_indices[:train_end]
    validation_indices = shuffled_indices[train_end:validation_end]
    test_indices = shuffled_indices[validation_end:]

    return MotionDatasetSplits(
        train_inputs=inputs[train_indices],
        train_targets=targets[train_indices],
        validation_inputs=inputs[validation_indices],
        validation_targets=targets[validation_indices],
        test_inputs=inputs[test_indices],
        test_targets=targets[test_indices],
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def fit_input_standardization(train_inputs: Tensor) -> StandardizationStats:
    """只使用训练输入计算逐特征均值和总体标准差。"""
    _validate_motion_inputs(train_inputs)
    if train_inputs.shape[0] < 2:
        raise ValueError("at least two training samples are required")

    mean = train_inputs.mean(dim=0)
    standard_deviation = train_inputs.std(dim=0, correction=0)
    if torch.any(standard_deviation == 0):
        raise ValueError("training inputs must vary in every feature")

    return StandardizationStats(
        mean=mean,
        standard_deviation=standard_deviation,
    )


def standardize_motion_inputs(
    inputs: Tensor,
    stats: StandardizationStats,
) -> Tensor:
    """使用训练集统计量标准化任意一组运动输入。"""
    _validate_motion_inputs(inputs)
    if stats.mean.shape != (4,) or stats.standard_deviation.shape != (4,):
        raise ValueError("standardization statistics must have shape (4,)")
    if torch.any(stats.standard_deviation <= 0):
        raise ValueError("standard deviations must be positive")

    return (inputs - stats.mean) / stats.standard_deviation


def _validate_motion_inputs(inputs: Tensor) -> None:
    """检查运动模型输入的基本形状和浮点类型。"""
    if inputs.ndim != 2 or inputs.shape[1] != 4:
        raise ValueError("inputs must have shape (N, 4)")
    if not inputs.is_floating_point():
        raise ValueError("inputs must use a floating-point dtype")
