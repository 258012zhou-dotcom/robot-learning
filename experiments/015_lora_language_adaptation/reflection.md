# 实验反思

## Checkpoint 工程补充

- adapter 只描述增量，不能独立替代冻结基础模型；新文件将两者完整保存，词表必须按原 Token ID 恢复，不能重新排序。
- 新加载器先按元数据重建架构并注入指定 LoRA 层，再严格加载所有参数和缓冲区，最后统一移动设备并进入评估模式。
- 本次验证使用临时目录与小模型；非零 LoRA 增量、全部冻结权重、输出一致性和实际推理入口均有回归检查。008/009 还人为让第二轮验证损失变差，检查保存的是第一轮最佳状态。
- 没有重跑完整训练，没有修改历史输出；这里的验证结论不增加模型准确率或真实机器人泛化证据，也不表示学习者已独立掌握 checkpoint。
- 文件用于推理恢复，不含优化器状态或随机数状态，不能用于逐步一致的断点续训。格式依赖当前架构代码，尚未承诺跨 PyTorch 版本或跨硬件逐位一致。
- 实际检查：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/zxd/miniconda3/envs/robot_learning/bin/python -m pytest -q tests/test_model_artifacts.py tests/test_object_detection.py tests/test_semantic_segmentation.py tests/test_peft.py`，结果为 `71 passed`。当前 PyTorch 为 `2.12.1+cu130`，CUDA 不可用，设备一致性已在 CPU 验证；CUDA 分支仅在可用时执行，本次未验证。

## 待验证问题

- 冻结模型是否无法直接理解未训练别名。
- LoRA 用少量参数能否接近完整微调的目标适配能力。
- 两种适配方法是否会损害原始规范描述能力。
- 参数效率与适配效果之间是否存在清晰权衡。

## 设计边界

- 图像分布不变，只改变语言描述，主要考查文本侧适配。
- LoRA 只注入文本 Transformer MLP，不代表这是所有 VLM 的最佳目标模块。
- 别名 Token 已在词表中但未参与来源训练，避免运行时调整词表和模型尺寸。
- 三次目标训练使用不同小样本，但共享基础模型、验证集和测试集。

## 实际结果

- Frozen 无法直接理解未训练别名，目标准确率保持在 `16.67%`。
- Rank 4 LoRA 三次目标适配均达到 `100.00%`，只训练 768 个参数。
- Full Fine-tuning 平均达到 `99.81% ± 0.26%`，目标能力与 LoRA 接近。
- LoRA 的来源准确率三次均为 `100.00%`，但来源 margin 有所下降，说明旧表示仍受到影响。
- Full Fine-tuning 的来源准确率为 `94.81% ± 5.06%`，其中第一组下降到 `87.78%`，小样本下波动和遗忘更明显。
- LoRA 与完整微调都很快达到 100% 验证准确率，但验证损失继续下降，说明后续训练主要增加分类间隔。

本实验的任务变化只发生在文本别名，且 LoRA 正好注入文本 MLP，因此结果对 LoRA 有利。若变化来自图像域、空间关系或跨模态融合层，同一注入位置可能失效。

## 工程问题

最初的通用 LoRA 包装器能通过普通 `Sequential` 测试，却不能直接运行 PyTorch Transformer 的评估快速路径。原因是框架会读取 `linear1.weight` 和 `linear1.bias`，而不是只调用包装层的 `forward()`。

修复后 `LoRALinear.weight` 动态返回合并权重 `W + scaling × B @ A`，使训练慢路径和评估快速路径使用同一个数学结果。新增回归测试确认非零 LoRA 在真实 VLM 评估中生效，并与合并模型输出一致。

## 下一步

- 根据结果分析 Rank、目标模块和学习率。
- 后续可比较 Adapter、Prompt Tuning 或 Hugging Face PEFT。
