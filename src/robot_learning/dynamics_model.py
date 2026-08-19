"""用于学习二维匀速运动关系的最小 PyTorch 模型。"""

import torch
from torch import Tensor, nn


class LearnedDynamicsModel(nn.Module):
    """从 x、y、vx、vy 预测下一时刻的二维位置。"""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=4, out_features=2)

    def forward(self, inputs: Tensor) -> Tensor:
        """执行批量预测，输入形状必须为 (N, 4)。"""
        if inputs.ndim != 2 or inputs.shape[1] != 4:
            raise ValueError("inputs must have shape (N, 4)")
        return self.linear(inputs)


def select_torch_device() -> torch.device:
    """优先选择 CUDA，没有可用 GPU 时回退到 CPU。"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
