# 实验 016：语言条件视觉目标定位

## 目标

给定一张包含多个物体的合成图像和一句目标指令，预测指令所描述物体的二维图像坐标，验证只有视觉与语言发生融合后才能解决条件定位问题。

## 输入与输出

- 图像：包含三个颜色与形状不同的物体。
- 指令：例如 `select the blue circle`。
- 输出：归一化目标中心坐标 `(x, y)`。

## 对比方法

- `fixed_center`：始终输出 `(0.5, 0.5)`，不使用输入。
- `vision_only`：查看图像，但不知道语言指定了哪个物体。
- `language_only`：读取指令，但看不到物体位于哪里。
- `multimodal`：使用语言 Query 在空间 Visual Tokens 上计算 Attention。

前三种方法用于证明单一模态的信息并不足以完成任务，而不是为了追求较高性能。

## 评价指标

- Mean Center Error：预测中心与目标中心的归一化欧氏距离，越低越好。
- Selection Accuracy：把预测坐标分配给最近候选物体后，是否选中了语言指定目标。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh \
  tests/test_multimodal_grounding.py \
  tests/test_multimodal_grounding_training.py
PYTHONPATH=src python experiments/016_multimodal_grounding/preview_data.py
PYTHONPATH=src python experiments/016_multimodal_grounding/run.py
python -m json.tool outputs/016_multimodal_grounding/results.json
```

## 输出

- `data_preview.png`：数据与目标标记。
- `learning_curves.png`：三种可训练方法的损失与验证误差。
- `multimodal_predictions.png`：Attention、真实目标和预测位置。
- `results.json`：四种方法的测试指标。
- `best_multimodal_model.pt`：验证集选择的多模态模型。
- `run.log`：实验日志。

## 实际结果

使用固定随机种子、1,200/240/300 个训练/验证/测试场景，在 RTX 5060 Laptop GPU（`cuda:0`）上运行：

| 方法 | 目标选择准确率 | 平均中心误差 |
| --- | ---: | ---: |
| 固定中心 | 35.67% | 0.3149 |
| 仅视觉 | 29.00% | 0.2830 |
| 仅语言 | 34.33% | 0.3176 |
| 视觉＋语言 | 100.00% | 0.0225 |

每个场景包含三个候选物体，因此前三个缺少必要信息的方法均处于约三选一的机会水平。多模态模型同时读取图像内容和目标指令后，在 300 个测试场景中全部选对目标，并将归一化中心误差降低到 `0.0225`。

这些结果证明了当前受控任务中的跨模态条件定位机制，但不能外推为真实图像、自然语言变化或机器人三维定位能力。
