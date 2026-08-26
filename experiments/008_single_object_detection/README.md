# 实验 008：单目标图像检测

## 目标

训练一个小型 CNN，在每张合成图像中同时判断目标类别并预测目标边界框，建立目标检测的最小完整训练闭环。

## 问题定义

- 图像中恰好包含一个目标：正方形或圆形。
- 分类头输出两个类别的 logits。
- 框回归头输出归一化的 `(center_x, center_y, width, height)`。
- 总损失为分类交叉熵与加权 Smooth L1 框损失之和。
- 检测成功要求类别正确且预测框 IoU 不低于 `0.5`。

## 对照基线

基线不读取图像，始终预测训练集中最常见类别和训练框的平均位置、平均大小。模型需要在分类准确率、平均 IoU 和联合检测成功率上与它比较。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_detection_geometry.py tests/test_object_detection.py
PYTHONPATH=src python experiments/008_single_object_detection/run.py
python -m json.tool outputs/008_single_object_detection/results.json
```

## 输出

- `outputs/008_single_object_detection/results.json`
- `outputs/008_single_object_detection/training_curve.png`
- `outputs/008_single_object_detection/detection_examples.png`
- `outputs/008_single_object_detection/run.log`

## 实际结果

在 RTX 5060 Laptop GPU（`cuda:0`）上使用固定种子 `42` 运行：

- 数据划分：训练 1400、验证 300、测试 300。
- 模型参数量：155558。
- 最佳验证轮次：第 35 轮。
- 无图像基线：分类准确率 47.33%，平均 IoU 0.1594，检测成功率 2.00%。
- CNN 检测器：分类准确率 100.00%，平均 IoU 0.8704，检测成功率 99.00%。

完整项目自动测试为 `169 passed`。预测框可视化中，大多数预测框与真值框明显重合。

## 当前边界

这是单目标检测学习基线，不包含多尺度候选框、objectness、多目标匹配或真实检测器中的完整 NMS 流程，结果也不能代表真实机器人视觉性能。
