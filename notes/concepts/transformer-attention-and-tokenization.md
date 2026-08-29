# Transformer、Attention 与 Tokenization

## 快速复习

Transformer 把不同来源的信息统一表示为 Token 序列：

```text
raw data
→ tokenization
→ content embedding + position embedding + modality embedding
→ Transformer blocks
→ task head
```

核心 Attention：

```text
Q = XWq, K = XWk, V = XWv
weights = softmax(QKᵀ / sqrt(dk), dim=-1)
output = weights V
```

Query 和 Key 决定读取谁，Value 决定实际汇总什么。原生 Self-Attention 只根据内容，不知道文字顺序、图像 Patch 位置或轨迹时间，因此通常必须加入位置表示。

## Tokenization 与 Embedding

Tokenization 决定一个信息单元代表什么；Embedding 把这个单元转换为统一维度 `D`：

```text
tokens.shape = (batch_size, token_count, embedding_dimension)
```

常见形式：

- 文本：子词 → Token ID → `nn.Embedding` 查表。
- 图像：图像 → Patch → 展平/卷积投影。
- 连续状态：关节角、速度、位姿 → Linear 或 MLP。
- 动作：连续向量投影，或量化为离散 Action Token。
- 轨迹：每个时间步或时间片作为 Token。

Token 设计会同时影响信息粒度、序列长度、Attention 成本、动作精度和多模态对齐方式。

## 图像 Patch Token

对于 `(N,C,H,W)` 图像和 Patch 大小 `P`：

```text
patch_count = (H/P) × (W/P)
patch_vector_dimension = C × P × P
```

使用：

```python
Conv2d(C, D, kernel_size=P, stride=P)
```

可以同时完成无重叠切块和线性投影，输出由 `(N,D,H/P,W/P)` 展平为 `(N,L,D)`。

实验代码已通过人工权重验证 Row-Major 顺序：四个 `2×2` Patch 的像素和依次成为 `[4,8,12,16]`，不是只验证输出形状。

Global Embedding 每张图只有一个向量，适合全局分类和检索；Patch Token 保留空间布局，更适合定位、多模态对齐和动作条件建模。

## Query、Key、Value

输入 `X` 经过三组可学习线性投影：

- Query：当前 Token 希望检索什么。
- Key：每个候选 Token 提供什么匹配索引。
- Value：候选 Token 被选中后传递什么内容。

若：

```text
Q: (N,Lq,Dk)
K: (N,Lk,Dk)
V: (N,Lk,Dv)
```

则：

```text
scores:  (N,Lq,Lk)
weights: (N,Lq,Lk)
output:  (N,Lq,Dv)
```

Softmax 必须沿 Key 维度进行，使每个 Query 分配给所有 Key 的权重和为 1。除以 `sqrt(Dk)` 用于避免高维点积过大、Softmax 过早饱和。

Attention 权重描述模型内部的信息路由，不应自动当作可靠的人类解释或因果证据。

## Self-Attention、Cross-Attention 与 Multi-Head

- Self-Attention：Q、K、V 来自同一序列，用于序列内部信息交换。
- Cross-Attention：Q 来自一个序列，K/V 来自另一个序列，用于视觉—语言、动作—观察等跨来源读取。
- Multi-Head Attention：把 `D` 拆成 `H` 个子空间，每个 Head 独立计算 Attention，再拼接并输出投影。

常见形状变化：

```text
(N,L,D)
→ Q/K/V projections
→ (N,H,L,D/H)
→ per-head attention
→ concatenate
→ (N,L,D)
```

通常要求 `D % H == 0`。多个 Head 提供不同子空间的建模能力，但不能保证每个 Head 自动对应清晰的人类概念。

## Transformer Encoder Block

项目实现 Pre-Norm Block：

```text
X = X + MultiHeadAttention(LayerNorm(X))
X = X + MLP(LayerNorm(X))
```

- Attention：在 Token 之间交换信息。
- MLP：独立变换每个 Token 内部特征。
- Residual：保留原始路径并改善深层优化。
- LayerNorm：稳定每个 Token 的特征尺度。

单元测试将 Attention 和 MLP 参数归零后，Block 输出严格等于输入，验证了残差路径的真实行为。梯度测试则确认输入、Q 投影和 MLP 都能收到梯度。

## 位置编码

没有位置时，Self-Attention 对排列等变；若后续使用平均或求和池化，整个序列模型会对排列不变：

```text
同一组 Token 的不同排列 → 相同序列级输出
```

常见输入组合：

```text
final token input
= content embedding
+ position embedding
+ modality/type embedding
```

常见位置方案：

- Learned Absolute Position Embedding：简单，但受最大长度限制。
- Sinusoidal Encoding：固定函数，可生成更长位置。
- Relative Position Bias：直接表示 Token 间相对距离。
- RoPE：把位置信息作用于 Q/K 的旋转关系。
- 二维/时空位置：用于图像行列和视频时间。

位置设计应匹配任务。语言需要顺序，图像需要二维空间，机器人轨迹需要时间；位置索引错误会让模型学习到错误关系。

## Mask

- Padding Mask：屏蔽补齐出来的无效 Token。
- Causal Mask：禁止当前位置读取未来 Token。
- 结构 Mask：限制特定模态或区域之间的连接。

项目接口明确规定 `True=允许读取`、`False=屏蔽`。不同框架的布尔语义可能相反，迁移代码时必须确认。

项目测试验证：因果 Mask 下所有未来权重严格为 0；修改未来 Token 后，较早位置的输出保持不变。

## 实验 012 的证据

任务判断 Token A 是否位于 Token B 之前。每个正负样本包含完全相同的 Token，只交换 A 与 B，且两个模型共有参数从相同初始值开始。

- 无位置编码：测试损失 `0.6932`，准确率 `50%`。
- 可学习位置编码：测试损失 `0.0004`，准确率 `100%`。
- 无位置模型对成对排列输出完全相同概率 `0.4926869`。
- 位置模型对正序和逆序输出 `0.9997968` 与 `0.0002189`。

`ln(2)≈0.6931` 对应平衡二分类的 50/50 预测。无位置模型停在这里不是普通训练失败，而是输入表示无法区分任务所需的顺序。位置编码为模型增加了必要信息。

## 具身智能中的连接

具身模型可能组合：

```text
[visual patch tokens]
[language instruction tokens]
[robot state tokens]
[action history tokens]
```

它们需要统一 Embedding 维度，同时保留：

- 图像二维位置。
- 指令词序。
- 状态与动作时间顺序。
- Token 的模态来源。

动作可连续回归，也可量化为离散 Token。离散化便于复用语言建模接口，但引入量化误差；连续输出更精确，但需要相应的概率分布和损失。

## 当前边界

当前已验证 Attention 数学、Multi-Head 形状、残差、因果 Mask、离散 Token、图像 Patch 和可学习绝对位置。实验 012 是固定长度合成序列，不代表自然语言理解、长上下文、视觉语言融合或机器人动作生成能力。
