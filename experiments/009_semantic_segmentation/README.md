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

- `outputs/009_semantic_segmentation/best_model.pt`（新训练才生成；本次修复未重跑历史完整训练）

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

## Checkpoint 与独立推理

新增 `best_model.pt` 保存验证损失最低的模型权重、架构名称与构造参数 `num_classes`、类别顺序、RGB 预处理、训练配置、最佳轮次与验证损失。训练函数恢复最佳状态后才保存，类别顺序为 `background, square, circle`，Mask 数字就是该顺序中的类别编号。

模型输入为 RGB、float32、NCHW，像素除以 255，不做均值/标准差归一化。入口复用共享 BGR→RGB 转换；图片尺寸必须等于 checkpoint 的 `image_size`，不自动拉伸或裁剪。加载使用 `weights_only=True`、`strict=True`、`eval()`，模型和输入统一放到 `--device`（默认 CPU），无需读取训练配置文件。架构固定层结构由仓库代码及格式版本 1 定义。

已有**新格式 checkpoint** 后，从仓库根目录执行以下示例。它只生成一张合成输入图并加载推理，不训练；如果尚无新 checkpoint，不能从历史指标或旧 adapter 凭空恢复模型。

```bash
conda activate robot_learning
PYTHONPATH=src python - <<'PY'
import cv2
import torch
from robot_learning.semantic_segmentation import generate_shape_segmentation_dataset
checkpoint = torch.load("outputs/009_semantic_segmentation/best_model.pt", map_location="cpu", weights_only=True)
size = checkpoint["preprocessing"]["image_size"]
image = generate_shape_segmentation_dataset(2, size, 123)[0][0]
rgb = image.mul(255).round().byte().permute(1, 2, 0).numpy()
cv2.imwrite("/tmp/009_checkpoint_input.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
PY
PYTHONPATH=src python experiments/009_semantic_segmentation/run.py --checkpoint outputs/009_semantic_segmentation/best_model.pt --image /tmp/009_checkpoint_input.png --device cpu
```

标准输出为 JSON，包含类别顺序与二维整数 Mask。替换 `--image` 可读取自己的同尺寸图片；合成任务模型的真实图像效果未验证。推理不会创建或改写训练输出。

验证命令：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_model_artifacts.py tests/test_object_detection.py tests/test_semantic_segmentation.py tests/test_peft.py`。新测试使用临时目录，包含权重/输出往返一致性、缺失/多余/形状错误权重的拒绝加载、三个独立进程推理入口，以及缩小数据规模的训练保存检查。微型测试通过不代表重新验证了历史训练分数。

## 当前边界

这是同分布合成数据上的语义分割基线，不涉及多实例区分、真实相机域差异、遮挡、深度信息或真实机器人部署。
