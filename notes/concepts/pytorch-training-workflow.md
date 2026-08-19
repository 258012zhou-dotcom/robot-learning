# PyTorch 训练、验证与评价流程

## 快速复习

- 训练集更新参数，验证集选择模型，测试集只用于最终评价。
- 归一化参数只能由训练集拟合，再应用到验证集和测试集。
- 标准训练步骤是 `zero_grad → forward → loss → backward → step`。
- 验证时同时使用 `model.eval()` 和 `torch.no_grad()`。
- 保存验证集表现最佳的 `state_dict`，而不是默认使用最后一轮。
- 损失用于优化；RMSE、成功率、延迟等指标用于解释任务表现。
- GPU 上的模型、输入和目标必须位于同一设备。

## 数据流

实验 004 使用输入 `[x, y, vx, vy]`，目标是固定时间间隔后的 `[next_x, next_y]`。数据首先用固定种子生成，再通过互不重叠的索引划分为训练、验证和测试集合。

```text
生成完整数据
  → 固定索引划分
  → 仅用训练输入拟合 mean/std
  → 标准化三个集合
  → 构建 DataLoader
```

验证和测试数据不能分别计算自己的均值和标准差，否则评价过程使用了它们自身的分布信息，形成数据泄漏。

## Tensor、批次与设备

运动输入形状为 `(N, 4)`，目标形状为 `(N, 2)`。DataLoader 将它们组织成 batch；数据保留在 CPU，取出 batch 后才移动到 `cuda:0`，避免一次把不必要的数据全部占用显存。

```python
inputs = inputs.to(device)
targets = targets.to(device)
```

测试代码应允许没有 GPU 时回退到 CPU；正式实验再记录实际设备和 CUDA 版本。

## 自动微分与参数更新

```python
optimizer.zero_grad(set_to_none=True)
predictions = model(inputs)
loss = loss_function(predictions, targets)
loss.backward()
optimizer.step()
```

`backward()` 计算并累积梯度，`step()` 才真正更新参数。`set_to_none=True` 可以避免不必要的梯度清零写入；下一次反向传播会重新创建梯度。

## 训练与验证模式

训练时使用 `model.train()`。验证时使用：

```python
model.eval()
with torch.no_grad():
    predictions = model(inputs)
```

`eval()` 控制 Dropout、BatchNorm 等层的行为，`no_grad()` 控制是否建立梯度计算图，二者作用不同。

平均损失应按 batch 的实际样本数量加权，避免最后一个较小 batch 与完整 batch 拥有相同权重。

## 最佳模型

每轮训练后计算验证损失。验证损失创新低时，应深复制当前 `state_dict`：

```python
best_state = deepcopy(model.state_dict())
```

如果只保留普通引用，其中的 Tensor 可能随后续训练继续变化。训练结束后恢复最佳参数，再执行测试集评价和保存 checkpoint。

checkpoint 至少保存：

- 模型参数
- 输入标准化参数
- 实验配置

只有模型权重而没有归一化参数，推理时就无法正确处理输入。

## 指标和基线

实验 004 同时比较：

- 不运动基线
- 学习模型
- 精确物理公式

结果表明学习模型恢复了接近 `position + velocity * 0.1` 的关系，但精确公式仍然更准确。基线让指标具有解释意义，也防止把“模型成功训练”误认为“机器学习是最佳方案”。

## 当前项目证据

- 数据划分、随机种子和标准化由 `motion_dataset.py` 实现。
- 模型和设备选择由 `dynamics_model.py` 实现。
- 训练、验证和最佳参数恢复由 `training.py` 实现。
- 实验 004 在 RTX 5060 Laptop GPU 上完成，测试 RMSE 约为 `6.59e-6`。
- 项目自动测试覆盖数据无重叠、标准化、梯度、设备和损失下降。

## 仍待验证

当前数据无噪声、关系线性且训练测试同分布。过拟合、正则化、分布变化和分类指标虽然已完成概念复习，但还没有在本项目中得到充分实验验证。
