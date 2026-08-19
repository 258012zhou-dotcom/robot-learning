"""训练一个最小 PyTorch 模型预测二维机器人的下一位置。"""

import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from robot_learning.dynamics_model import (
    LearnedDynamicsModel,
    select_torch_device,
)
from robot_learning.motion_dataset import (
    fit_input_standardization,
    generate_motion_samples,
    split_motion_dataset,
    standardize_motion_inputs,
)
from robot_learning.training import (
    create_data_loader,
    evaluate_mse,
    train_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "004_learned_dynamics.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "004_learned_dynamics"


def load_config() -> dict[str, Any]:
    """读取实验配置。"""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def set_random_seeds(seed: int) -> None:
    """设置本实验使用的 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_rmse(predictions: Tensor, targets: Tensor) -> float:
    """计算所有样本和坐标维度上的 RMSE。"""
    return float(torch.sqrt(torch.mean((predictions - targets) ** 2)).item())


def recover_raw_linear_parameters(
    model: LearnedDynamicsModel,
    input_mean: Tensor,
    input_standard_deviation: Tensor,
) -> tuple[Tensor, Tensor]:
    """把标准化输入空间中的线性参数转换回原始输入尺度。"""
    weight = model.linear.weight.detach().cpu()
    bias = model.linear.bias.detach().cpu()
    raw_weight = weight / input_standard_deviation
    raw_bias = bias - raw_weight @ input_mean
    return raw_weight, raw_bias


def main() -> None:
    """运行数据生成、训练、测试和结果保存流程。"""
    config = load_config()
    seed = int(config["seed"])
    set_random_seeds(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "run.log", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)

    inputs, targets = generate_motion_samples(
        sample_count=int(config["sample_count"]),
        dt=float(config["dt"]),
        seed=seed,
    )
    splits = split_motion_dataset(
        inputs,
        targets,
        train_fraction=float(config["train_fraction"]),
        validation_fraction=float(config["validation_fraction"]),
        seed=seed,
    )
    stats = fit_input_standardization(splits.train_inputs)

    standardized_train_inputs = standardize_motion_inputs(
        splits.train_inputs,
        stats,
    )
    standardized_validation_inputs = standardize_motion_inputs(
        splits.validation_inputs,
        stats,
    )
    standardized_test_inputs = standardize_motion_inputs(
        splits.test_inputs,
        stats,
    )

    batch_size = int(config["batch_size"])
    train_loader = create_data_loader(
        standardized_train_inputs,
        splits.train_targets,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = create_data_loader(
        standardized_validation_inputs,
        splits.validation_targets,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = create_data_loader(
        standardized_test_inputs,
        splits.test_targets,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    device = select_torch_device()
    model = LearnedDynamicsModel()
    training_result = train_model(
        model,
        train_loader,
        validation_loader,
        device=device,
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    learned_test_mse = evaluate_mse(model, test_loader, device)
    learned_test_rmse = math.sqrt(learned_test_mse)

    no_motion_predictions = splits.test_inputs[:, :2]
    no_motion_rmse = calculate_rmse(
        no_motion_predictions,
        splits.test_targets,
    )
    physics_predictions = (
        splits.test_inputs[:, :2]
        + splits.test_inputs[:, 2:] * float(config["dt"])
    )
    physics_rmse = calculate_rmse(
        physics_predictions,
        splits.test_targets,
    )

    raw_weight, raw_bias = recover_raw_linear_parameters(
        model,
        stats.mean,
        stats.standard_deviation,
    )
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )

    checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "input_mean": stats.mean,
        "input_standard_deviation": stats.standard_deviation,
        "config": config,
    }
    torch.save(checkpoint, OUTPUT_DIR / "best_model.pt")

    results = {
        "experiment_name": config["experiment_name"],
        "seed": seed,
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "split_sizes": {
            "train": len(splits.train_indices),
            "validation": len(splits.validation_indices),
            "test": len(splits.test_indices),
        },
        "best_epoch": training_result.best_epoch,
        "best_validation_mse": training_result.best_validation_loss,
        "learned_test_rmse": learned_test_rmse,
        "no_motion_test_rmse": no_motion_rmse,
        "physics_test_rmse": physics_rmse,
        "raw_scale_weight": raw_weight.tolist(),
        "raw_scale_bias": raw_bias.tolist(),
    }
    with (OUTPUT_DIR / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    figure, axis = plt.subplots(figsize=(8, 4))
    epochs = np.arange(1, len(training_result.training_losses) + 1)
    axis.plot(
        epochs,
        training_result.training_losses,
        label="training MSE",
    )
    axis.plot(
        epochs,
        training_result.validation_losses,
        label="validation MSE",
    )
    axis.axvline(
        training_result.best_epoch,
        color="gray",
        linestyle="--",
        label="best epoch",
    )
    axis.set(
        title="Learned Dynamics Training Curve",
        xlabel="epoch",
        ylabel="MSE",
        yscale="log",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
    plt.close(figure)

    logger.info("设备：%s (%s)", device, device_name)
    logger.info("数据划分：%s", results["split_sizes"])
    logger.info("最佳轮次：%s", training_result.best_epoch)
    logger.info("最佳验证 MSE：%.8f", training_result.best_validation_loss)
    logger.info("学习模型测试 RMSE：%.8f", learned_test_rmse)
    logger.info("不运动基线测试 RMSE：%.8f", no_motion_rmse)
    logger.info("物理公式测试 RMSE：%.8f", physics_rmse)
    logger.info("结果目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
