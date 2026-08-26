# 实验 007：CNN 合成图像分类

## 目标

训练一个小型卷积神经网络区分方形与圆形，验证图像数据生成、数据划分、Batch 训练、验证集模型选择、测试集评价和失败分析的完整流程。

本实验用于理解 CNN 分类系统，不代表真实机器人视觉性能。

## 数据

- 类别 `0`：方形（square）。
- 类别 `1`：圆形（circle）。
- 图像为 RGB、`float32`、NCHW，值域 `[0,1]`。
- 图形的颜色、位置、大小和背景噪声随机变化。
- 随机种子固定，结果可以重复生成。
- 训练集、验证集和测试集由互不重叠的随机索引划分。

训练前先查看 `dataset_preview.png`，检查标签并排除明显的颜色、位置等类别捷径。

训练完成后还会评价一个不参与训练和模型选择的 OOD 挑战集。背景、颜色、尺寸和位置分布保持不变，只把方形旋转到训练中未见过的角度，用受控变量检查旋转泛化。

## 模型

```text
Conv(3→16) + ReLU + MaxPool
→ Conv(16→32) + ReLU + MaxPool
→ Conv(32→64) + ReLU
→ AdaptiveAvgPool(1×1)
→ Linear(64→2)
```

模型输出两个 Logits，训练使用 `CrossEntropyLoss` 和 AdamW。每轮在验证集上评价，并在训练结束后恢复验证损失最低的模型参数。

实验训练两个结构与超参数相同的模型：Baseline 使用原始训练集；Augmented 模型使用数量不变、但在合成渲染阶段随机改变部分方形角度的训练集。这属于生成器级领域随机化，不会引入整帧旋转造成的插值边缘和目标裁切。

## 运行

在项目根目录执行：

```bash
conda activate robot_learning
./scripts/run_tests.sh tests/test_image_classification.py
PYTHONPATH=src python experiments/007_cnn_classification/preview_data.py
PYTHONPATH=src python experiments/007_cnn_classification/run.py
python -m json.tool outputs/007_cnn_classification/results.json
```

## 输出

结果位于 `outputs/007_cnn_classification/`：

- `dataset_preview.png`：训练前数据检查。
- `learning_curves.png`：训练和验证交叉熵曲线。
- `augmentation_preview.png`：生成器级旋转随机化训练样本。
- `augmented_learning_curves.png`：旋转增强模型学习曲线。
- `confusion_matrix.png`：测试集混淆矩阵，行是真实类别，列是预测类别。
- `misclassified_examples.png`：最多 12 个测试错误样本。
- `challenge_preview.png`：未见角度方形组成的 OOD 挑战集样本。
- `challenge_confusion_matrix.png`：OOD 挑战集混淆矩阵。
- `challenge_misclassified_examples.png`：OOD 挑战集错误样本。
- `augmented_challenge_confusion_matrix.png`：增强模型的挑战集混淆矩阵。
- `augmented_challenge_misclassified_examples.png`：增强模型的挑战错误样本。
- `best_model.pt`：最佳模型参数、类别名称和实验配置。
- `best_augmented_model.pt`：旋转增强模型的最佳参数。
- `results.json`：设备、数据划分、最佳轮次、基线和测试指标。
- `run.log`：运行日志。

生成结果不提交 Git。
