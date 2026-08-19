"""机器人位置预测的评价指标与分组误差分析。"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ErrorSummary:
    """整体坐标 RMSE 和逐样本欧氏误差统计。"""

    coordinate_rmse: float
    median_euclidean_error: float
    p95_euclidean_error: float
    maximum_euclidean_error: float


@dataclass(frozen=True)
class SpeedGroupErrors:
    """按测试样本速度中位数划分的两组 RMSE。"""

    speed_threshold: float
    low_speed_rmse: float
    high_speed_rmse: float
    low_speed_count: int
    high_speed_count: int


def euclidean_errors(predictions: Tensor, targets: Tensor) -> Tensor:
    """返回每个二维位置预测的欧氏距离误差。"""
    _validate_predictions(predictions, targets)
    return torch.linalg.vector_norm(predictions - targets, dim=1)


def summarize_prediction_errors(
    predictions: Tensor,
    targets: Tensor,
) -> ErrorSummary:
    """汇总坐标 RMSE 以及逐样本误差的中位数、P95 和最大值。"""
    errors = euclidean_errors(predictions, targets)
    coordinate_rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))

    return ErrorSummary(
        coordinate_rmse=float(coordinate_rmse.item()),
        median_euclidean_error=float(torch.median(errors).item()),
        p95_euclidean_error=float(torch.quantile(errors, 0.95).item()),
        maximum_euclidean_error=float(torch.max(errors).item()),
    )


def summarize_errors_by_speed(
    predictions: Tensor,
    targets: Tensor,
    velocities: Tensor,
) -> SpeedGroupErrors:
    """按速度中位数将样本分为低速组和高速组并计算坐标 RMSE。"""
    _validate_predictions(predictions, targets)
    if velocities.ndim != 2 or velocities.shape != predictions.shape:
        raise ValueError("velocities must have shape (N, 2)")

    speeds = torch.linalg.vector_norm(velocities, dim=1)
    threshold = torch.median(speeds)
    low_speed_mask = speeds <= threshold
    high_speed_mask = speeds > threshold
    if not torch.any(low_speed_mask) or not torch.any(high_speed_mask):
        raise ValueError("speed groups must both contain at least one sample")

    low_speed_rmse = torch.sqrt(
        torch.mean(
            (predictions[low_speed_mask] - targets[low_speed_mask]) ** 2
        )
    )
    high_speed_rmse = torch.sqrt(
        torch.mean(
            (predictions[high_speed_mask] - targets[high_speed_mask]) ** 2
        )
    )

    return SpeedGroupErrors(
        speed_threshold=float(threshold.item()),
        low_speed_rmse=float(low_speed_rmse.item()),
        high_speed_rmse=float(high_speed_rmse.item()),
        low_speed_count=int(low_speed_mask.sum().item()),
        high_speed_count=int(high_speed_mask.sum().item()),
    )


def _validate_predictions(predictions: Tensor, targets: Tensor) -> None:
    """检查二维位置预测和目标的形状。"""
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("predictions must have shape (N, 2)")
    if targets.shape != predictions.shape:
        raise ValueError("targets must have the same shape as predictions")
    if predictions.shape[0] == 0:
        raise ValueError("at least one prediction is required")
