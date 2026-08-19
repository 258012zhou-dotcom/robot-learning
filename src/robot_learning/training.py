"""小型 PyTorch 实验复用的训练与验证流程。"""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingResult:
    """训练曲线以及验证集上表现最佳的轮次。"""

    training_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    best_validation_loss: float


def create_data_loader(
    inputs: Tensor,
    targets: Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """从内存张量建立可复现的 DataLoader。"""
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError("inputs and targets must be two-dimensional")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must contain the same sample count")
    if inputs.shape[0] == 0:
        raise ValueError("data loader requires at least one sample")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        TensorDataset(inputs, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    """训练一轮并返回按样本数量加权的平均 MSE。"""
    model.train()
    loss_function = nn.MSELoss()
    total_loss = 0.0
    sample_count = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return total_loss / sample_count


def evaluate_mse(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """在不记录梯度的情况下计算按样本加权的平均 MSE。"""
    model.eval()
    loss_function = nn.MSELoss()
    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = loss_function(model(inputs), targets)

            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            sample_count += batch_size

    return total_loss / sample_count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainingResult:
    """训练模型，并在结束时恢复验证损失最低的参数。"""
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    training_losses: list[float] = []
    validation_losses: list[float] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state = deepcopy(model.state_dict())

    for epoch_index in range(epochs):
        training_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )
        validation_loss = evaluate_mse(
            model,
            validation_loader,
            device,
        )
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch_index + 1
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return TrainingResult(
        training_losses=training_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )
