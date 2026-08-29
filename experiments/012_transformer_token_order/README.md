# 实验 012：位置编码与 Token 顺序

## 目标

通过严格配对的 Token 序列验证：Self-Attention 只处理内容时无法识别顺序，加入位置编码后才能学习“A 是否出现在 B 前面”。

## 对照设计

每个正样本都有一个包含完全相同 Token 的负样本，只交换特殊 Token A=`1` 和 B=`2`：

```text
[A, x, y, B, z] → 1
[B, x, y, A, z] → 0
```

两种模型使用相同的 Token Embedding、两层 Transformer、平均池化和分类头。共有参数从完全相同的初始权重开始，唯一额外参数是 Learned Position Embedding。

无位置模型在平均池化后对排列不变，因此同一对样本得到相同 logits，理论上无法超过平衡数据的 50% 准确率。位置模型可以把 Token 内容与绝对位置结合。

## 验证层次

- 单元测试：证明配对数据契约、排列不变性、梯度链路和输出形状。
- 训练对照：比较两种模型的训练曲线与独立测试集准确率。
- 成对输出：检查同一正负样本的正类概率是否能够分离。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_attention.py tests/test_transformer_tokens.py tests/test_sequence_transformer.py
PYTHONPATH=src python experiments/012_transformer_token_order/run.py
python -m json.tool outputs/012_transformer_token_order/results.json
```

## 输出

- `outputs/012_transformer_token_order/results.json`
- `outputs/012_transformer_token_order/learning_comparison.png`
- `outputs/012_transformer_token_order/test_accuracy.png`
- `outputs/012_transformer_token_order/run.log`

## 实际结果

使用 RTX 5060 Laptop GPU、固定种子 `42` 运行：

- 无位置编码：最佳轮次 17，测试损失 `0.6932`，测试准确率 `50.00%`。
- 可学习位置编码：最佳轮次 25，测试损失 `0.0004`，测试准确率 `100.00%`。
- 第一组成对样本在无位置模型中的正类概率完全相同，均为 `0.4926869`。
- 第一组成对样本在位置模型中的正类概率分别为 `0.9997968` 和 `0.0002189`。

无位置模型的训练损失始终接近 `ln(2)`，验证准确率始终为平衡二分类的 50%。位置模型约从第 7 轮开始明显学习顺序关系，并在第 11 轮达到 100% 验证准确率。

该结果说明失败原因不是模型层数不足或训练时间不够，而是无位置输入对成对排列产生完全相同的序列级表示。加入位置向量后，模型才具有区分顺序所需的信息。

## 当前边界

这是离散合成序列上的受控实验，用于隔离位置编码的作用，不代表语言理解或真实机器人时序建模能力。
