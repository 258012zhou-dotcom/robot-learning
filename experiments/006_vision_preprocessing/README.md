# 实验 006：视觉预处理与坐标同步

## 目标

建立一个接近具身智能视觉输入端的最小可复现流水线：

```text
BGR uint8 相机图像
→ HSV 可解释基线产生目标框
→ Letterbox
→ 同步变换目标框和相机内参
→ RGB float32 NCHW 模型张量
```

HSV 目标提取只是用于验证流水线的简单基线。后续 CNN、目标检测器或视觉语言模型可以替换它，而图像契约、几何变换、相机内参和张量转换仍可复用。

## 合成场景

- 原始图像：`640 × 480`，OpenCV BGR、`uint8`、HWC。
- 主要绿色目标框：`(400, 120, 120, 160)`。
- 场景还包含一个较小绿色区域和一个红色干扰物。
- 模型输入：`224 × 224`。
- 原始相机内参：`fx=fy=600`、`cx=319.5`、`cy=239.5`。

理论上，Letterbox 缩放率为 `0.35`，上下各填充 `28` 像素。因此主要目标框应映射为 `(140, 70, 42, 56)`。

## 运行

在项目根目录执行：

```bash
conda activate robot_learning
./scripts/run_tests.sh tests/test_vision_preprocessing.py
PYTHONPATH=src python experiments/006_vision_preprocessing/run.py
python -m json.tool outputs/006_vision_preprocessing/results.json
```

## 输出

结果位于 `outputs/006_vision_preprocessing/`：

- `scene.png`：原始合成相机场景。
- `mask.png`：HSV 二值目标 Mask。
- `annotated_original.png`：原图检测框。
- `letterboxed.png`：保持宽高比并补边后的模型输入图像。
- `annotated_model_input.png`：Letterbox 坐标系中的检测框。
- `results.json`：目标框、相机内参、变换参数和模型张量契约。

这些文件可以由配置和代码重新生成，因此不提交 Git。
