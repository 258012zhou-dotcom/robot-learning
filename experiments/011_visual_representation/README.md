# 实验 011：视觉表示学习

## 目标

比较相同结构、相同初始化下的三种视觉编码器：

- 未训练编码器：提供随机特征基线。
- 监督编码器：用正方形/圆形标签和交叉熵训练。
- 对比编码器：不读取标签，只让同一图像的两种增强视图接近。

训练后统一冻结编码器，使用训练集特征作为参考库，通过余弦相似度 1-NN 评估测试图像。

## 公平比较

- 三种编码器使用相同网络结构和初始权重。
- 监督学习和对比学习只接触训练集。
- 测试时统一移除监督分类头。
- 三种表示使用相同的 1-NN 分类规则。
- 同时评估干净测试集和颜色/噪声外观扰动测试集。

## 关键指标

- `clean_test_1nn_accuracy`：干净测试图像的冻结特征 1-NN 准确率。
- `appearance_challenge_1nn_accuracy`：外观扰动图像的 1-NN 准确率。
- `clean_challenge_mean_cosine_similarity`：同一图像扰动前后特征的平均余弦相似度。
- `mean_dimension_standard_deviation`：各特征维度是否保留变化。
- `mean_pairwise_cosine_similarity`：不同样本是否过度相似，用于辅助观察表示坍塌。

单独看训练损失下降不能证明表示有用，必须结合冻结特征评估与坍塌诊断。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_visual_representation.py
PYTHONPATH=src python experiments/011_visual_representation/run.py
python -m json.tool outputs/011_visual_representation/results.json
```

## 输出

- `outputs/011_visual_representation/results.json`
- `outputs/011_visual_representation/training_curves.png`
- `outputs/011_visual_representation/accuracy_comparison.png`
- `outputs/011_visual_representation/embedding_comparison.png`
- `outputs/011_visual_representation/augmentation_preview.png`
- `outputs/011_visual_representation/run.log`

## 实际结果

使用 RTX 5060 Laptop GPU、固定种子 `42` 运行：

- 未训练编码器：干净集 1-NN 为 `63.33%`，外观挑战为 `52.78%`。
- 监督编码器：干净集 1-NN 为 `100.00%`，外观挑战为 `63.33%`。
- 对比编码器：干净集 1-NN 为 `100.00%`，外观挑战为 `73.33%`。
- 同一图像扰动前后的平均余弦相似度分别为 `0.9990`、`0.2693` 和 `0.6783`。

未训练编码器的高相似度不是最强鲁棒性：其平均维度标准差只有 `0.0065`，不同样本的平均余弦相似度达到 `0.9978`，说明大多数图像都被映射到非常接近的方向。监督与对比表示的维度标准差分别为 `0.1675` 和 `0.1581`，保留了明显更多的样本变化。

二维投影中，未训练特征的两类高度混杂；监督表示形成两个紧密类别方向；对比表示形成较展开但仍可分离的两类结构。训练曲线均持续下降，没有数值发散。

## 当前边界

这是两类合成图像上的小型对比学习实验，不代表真实机器人图像上的预训练效果。增强只包含颜色增益和噪声，没有随机裁剪、几何变化、投影头或大规模负样本。
