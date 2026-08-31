# 实验 014：视觉语言表示对齐

## 目标

使用一个小型 CLIP 风格双编码器，把合成图像和英文短描述映射到同一个向量空间，验证图文对比学习、语义检索和无固定分类头的提示词分类机制。

## 数据与方法

- 视觉概念：`red/green/blue × square/circle`，共 6 类。
- 图像侧：小型 CNN 输出归一化图像向量。
- 文本侧：词嵌入、位置编码和小型 Transformer 输出归一化文本向量。
- 训练侧：每个 batch 从每个概念各取一张图，使用对称 Image→Text 与 Text→Image 对比损失。
- 验证与测试：每个概念只使用一条规范文本，但允许对应多张同概念图像，按语义而不是固定实例评价。

## 评价指标

- Image→Text Accuracy：每张图像是否选择了正确概念文本。
- Text→Image Accuracy：每条文本的最高分图像是否属于正确概念。
- Mean Similarity Margin：正确文本分数减去最强错误文本分数的平均值。
- Confusion Matrix：图像真实概念与预测文本概念之间的错误结构。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_vision_language.py
PYTHONPATH=src python experiments/014_vision_language_alignment/run.py
python -m json.tool outputs/014_vision_language_alignment/results.json
```

## 输出

- `outputs/014_vision_language_alignment/results.json`
- `outputs/014_vision_language_alignment/best_model.pt`
- `outputs/014_vision_language_alignment/learning_curves.png`
- `outputs/014_vision_language_alignment/similarity_matrices.png`
- `outputs/014_vision_language_alignment/confusion_matrices.png`
- `outputs/014_vision_language_alignment/run.log`

## 实际结果

使用 RTX 5060 Laptop GPU、固定随机种子和独立生成的训练、验证、测试集运行：

- 训练集、验证集和测试集分别包含 300、90、180 张图像。
- 模型参数量为 `34,624`，最佳验证轮次为第 29 轮。
- 未训练模型的 Image→Text 与 Text→Image 准确率均为 `16.67%`，等于六分类随机水平。
- 未训练平均 similarity margin 为 `-1.4253`，说明正确文本通常不占优势。
- 训练后双向准确率均为 `100.00%`。
- 训练后平均 similarity margin 为 `7.3462`，代表正确文本明显超过最强错误文本。
- 未训练混淆矩阵把所有图像都预测为 `a blue square`；训练后 180 张测试图像全部落在正确对角线上。

本结果证明模型在当前封闭的六概念分布中学会了图文对齐。因为六个概念及其规范描述都参与过训练，所以这不是对未见类别的严格 Zero-shot 证据。

## 当前边界

这是受控合成数据上的概念对齐实验，不包含未见类别、自然语言同义表达、真实图像、开放词表、大规模预训练或视觉问答，不能代表真实 VLM 的 Zero-shot 和开放世界能力。
