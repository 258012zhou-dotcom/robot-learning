# 实验 015：LoRA 语言别名适配

## 目标

在同一个视觉语言双编码器上比较 Frozen、LoRA 和 Full Fine-tuning，观察少量新语言描述的适配能力、原语言保持率和可训练参数量。

## 任务

基础模型先学习规范描述，例如：

- `a red square`
- `a green circle`

目标适配把同一视觉概念改写成训练前未使用的别名，例如：

- `crimson box`
- `emerald disk`
- `azure box`

别名词提前存在于词表中，但在来源预训练中没有出现，因此对应 Token Embedding 没有得到语义训练。每次目标适配只有 30 对图文，重复 3 组不同小样本。

## 对照方法

- `frozen`：基础模型完全冻结，不执行目标适配。
- `lora`：只在文本 Transformer 的 `linear1` 和 `linear2` 中训练 Rank 4 LoRA。
- `full_finetune`：更新全部模型参数。

每种方法都在目标别名测试集上评价适配能力，并在原始规范描述测试集上重新评价来源能力。

## 运行

```bash
conda activate robot_learning
cd ~/AI_Project/robot-learning
./scripts/run_tests.sh tests/test_peft.py
PYTHONPATH=src python experiments/015_lora_language_adaptation/run.py
python -m json.tool outputs/015_lora_language_adaptation/results.json
```

## 输出

- `outputs/015_lora_language_adaptation/results.json`
- `outputs/015_lora_language_adaptation/trial_1_lora_adapter.pt`
- `outputs/015_lora_language_adaptation/accuracy_comparison.png`
- `outputs/015_lora_language_adaptation/trainable_parameters.png`
- `outputs/015_lora_language_adaptation/first_trial_learning_curves.png`
- `outputs/015_lora_language_adaptation/run.log`

## 实际结果

在 RTX 5060 Laptop GPU 上固定基础模型，并使用三组不同的每类 5 张目标训练图运行：

- 基础模型包含 `34,784` 个参数，来源规范描述测试准确率为 `100.00%`。
- Frozen 的目标别名准确率为 `16.67% ± 0.00%`，等于六分类随机水平。
- LoRA 只训练 `768` 个参数，占加入 LoRA 后存储参数的约 `2.16%`。
- LoRA 目标别名准确率为 `100.00% ± 0.00%`，来源规范描述准确率保持 `100.00% ± 0.00%`。
- Full Fine-tuning 训练全部 `34,784` 个参数，目标准确率为 `99.81% ± 0.26%`。
- Full Fine-tuning 的来源准确率下降到 `94.81% ± 5.06%`；第一组试验只有 `87.78%`，表现出更明显的来源能力波动。

LoRA 的来源准确率没有下降，但来源 mean similarity margin 从基础模型的 `7.0357` 降到三次试验的约 `4.20–5.40`，说明表示仍然发生了改变，不能解释为完全没有遗忘。

这个结果说明：当任务变化主要位于文本 Transformer MLP 可以表达的语言映射时，选对注入位置的 LoRA 可以用很少参数完成有效适配，并比当前完整微调更稳定地保持来源能力。这不是 LoRA 在所有任务上都优于完整微调的证据。

## 当前边界

这是小型合成模型上的语言映射实验。LoRA 的计算、显存和存储优势在真实大模型上更明显；当前结果不能直接外推到大型 VLM 或 LLM。
