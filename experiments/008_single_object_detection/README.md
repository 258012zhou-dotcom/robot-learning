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

- `outputs/008_single_object_detection/best_model.pt`（新训练才生成；本次修复未重跑历史完整训练）

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

## Checkpoint 与独立推理

新增 `best_model.pt` 保存验证损失最低的模型权重、架构名称与构造参数 `num_classes`、类别顺序、RGB 预处理、训练配置、最佳轮次与验证损失。训练函数恢复最佳状态后才保存，类别顺序为 `square, circle`，框格式为归一化 `(center_x, center_y, width, height)`。

模型输入为 RGB、float32、NCHW，像素除以 255，不做均值/标准差归一化。入口复用共享 BGR→RGB 转换；图片尺寸必须等于 checkpoint 的 `image_size`，不自动拉伸或裁剪。加载使用 `weights_only=True`、`strict=True`、`eval()`，模型和输入统一放到 `--device`（默认 CPU），无需读取训练配置文件。架构固定层结构由仓库代码及格式版本 1 定义。

已有**新格式 checkpoint** 后，从仓库根目录执行以下示例。它只生成一张合成输入图并加载推理，不训练；如果尚无新 checkpoint，不能从历史指标或旧 adapter 凭空恢复模型。

```bash
conda activate robot_learning
PYTHONPATH=src python - <<'PY'
import cv2
import torch
from robot_learning.object_detection import generate_shape_detection_dataset
checkpoint = torch.load("outputs/008_single_object_detection/best_model.pt", map_location="cpu", weights_only=True)
size = checkpoint["preprocessing"]["image_size"]
image = generate_shape_detection_dataset(2, size, 123)[0][0]
rgb = image.mul(255).round().byte().permute(1, 2, 0).numpy()
cv2.imwrite("/tmp/008_checkpoint_input.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
PY
PYTHONPATH=src python experiments/008_single_object_detection/run.py --checkpoint outputs/008_single_object_detection/best_model.pt --image /tmp/008_checkpoint_input.png --device cpu
```

标准输出为 JSON，包含类别与归一化预测框。替换 `--image` 可读取自己的同尺寸图片；合成任务模型的真实图像效果未验证。推理不会创建或改写训练输出。

验证命令：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_model_artifacts.py tests/test_object_detection.py tests/test_semantic_segmentation.py tests/test_peft.py`。新测试使用临时目录，包含权重/输出往返一致性、缺失/多余/形状错误权重的拒绝加载、三个独立进程推理入口，以及缩小数据规模的训练保存检查。微型测试通过不代表重新验证了历史训练分数。

## 当前边界

这是单目标检测学习基线，不包含多尺度候选框、objectness、多目标匹配或真实检测器中的完整 NMS 流程，结果也不能代表真实机器人视觉性能。
