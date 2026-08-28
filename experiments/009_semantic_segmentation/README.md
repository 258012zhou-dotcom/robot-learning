# 实验 009：合成图像语义分割

## 目标

训练一个小型 U-Net 风格网络，对合成图像中的背景、正方形和圆形进行像素级分类，建立语义分割的数据、训练、评价和可视化闭环。

## 问题定义

- 输入为归一化 RGB Tensor，形状为 `(N, 3, H, W)`。
- 标签为整数类别 Mask，形状为 `(N, H, W)`。
- 类别 0 为背景，类别 1 为正方形，类别 2 为圆形。
- 模型输出每个像素的三个类别 logits。
- 损失为带逆频率类别权重的像素级 Cross Entropy。

## 评价

- Pixel Accuracy。
- 背景、正方形和圆形的各类别 IoU。
- 三类别 mIoU。
- 只统计正方形和圆形的 Foreground mIoU。
- 全背景预测基线。

全背景基线用于证明：背景占比高时，Pixel Accuracy 可能很高，但模型仍可能完全没有识别前景目标。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_semantic_segmentation.py
PYTHONPATH=src python experiments/009_semantic_segmentation/run.py
python -m json.tool outputs/009_semantic_segmentation/results.json
```

## 输出

- `outputs/009_semantic_segmentation/results.json`
- `outputs/009_semantic_segmentation/training_curve.png`
- `outputs/009_semantic_segmentation/confusion_matrix.png`
- `outputs/009_semantic_segmentation/segmentation_examples.png`
- `outputs/009_semantic_segmentation/run.log`

## 实际结果

在 RTX 5060 Laptop GPU（`cuda:0`）上使用固定种子 `42` 运行：

- 数据划分：训练 840、验证 180、测试 180。
- 模型参数量：118307。
- 最佳验证轮次：第 18 轮。
- 全背景基线：Pixel Accuracy 88.65%，mIoU 0.2955，Foreground mIoU 0。
- 分割模型：Pixel Accuracy 99.93%，mIoU 0.9950，Foreground mIoU 0.9929。
- 类别 IoU：背景 0.9992、正方形 0.9992、圆形 0.9867。

预测 Mask 与真值 Mask 在可视化中基本重合。圆形边界的 IoU 略低于正方形，符合离散像素曲线边界更难完全重合的特点。

## 当前边界

这是同分布合成数据上的语义分割基线，不涉及多实例区分、真实相机域差异、遮挡、深度信息或真实机器人部署。
