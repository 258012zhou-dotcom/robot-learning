# 实验 004：PyTorch 学习二维运动模型

## 目标

用一个只有 10 个参数的线性模型，从当前位置和速度预测固定时间间隔后的二维位置。该实验用于一次性验证阶段 1 的机器学习基础流程，不是为了取代已知且准确的物理公式。

输入为 `[x, y, vx, vy]`，目标为：

```text
next_position = position + velocity * dt
```

## 数据与划分

程序根据固定随机种子生成二维位置和速度，并按原始样本索引划分：

- 70% 训练集
- 15% 验证集
- 15% 测试集

只有训练输入用于计算均值和标准差。验证集和测试集使用同一组训练统计量，避免数据泄漏。

## 模型与训练

模型为 `Linear(4, 2)`，使用 MSE、AdamW 和验证集最佳模型恢复。训练优先使用 `cuda:0`，没有 CUDA 时自动回退到 CPU。

## 对比方法

- 不运动基线：直接把当前位置作为下一位置。
- 学习模型：从样本中学习运动关系。
- 物理公式：使用已知的精确匀速公式。

物理公式应当最好；学习模型应明显优于不运动基线，并接近物理公式。

## 运行

在项目根目录执行：

```bash
conda activate robot_learning
./scripts/run_tests.sh
PYTHONPATH=src python experiments/004_learned_dynamics/run.py
```

## 输出

生成文件位于 `outputs/004_learned_dynamics/`：

- `results.json`：设备、划分、指标和原始尺度模型参数。
- `best_model.pt`：最佳模型参数、训练集标准化参数和配置。
- `loss_curve.png`：训练与验证损失曲线。
- `test_error_distribution.png`：三种方法的测试集逐样本欧氏误差分布。
- `run.log`：运行日志。

这些输出由配置和代码重新生成，不提交 Git。
