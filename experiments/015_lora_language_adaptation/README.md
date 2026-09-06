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

- `outputs/015_lora_language_adaptation/trial_1_lora_model.pt`（新训练才生成；本次修复未重跑历史完整训练）

- `outputs/015_lora_language_adaptation/results.json`
- `outputs/015_lora_language_adaptation/trial_1_lora_adapter.pt`（历史产物，新训练不再生成）
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

## Checkpoint 与独立推理

新增 `trial_1_lora_model.pt` 保存第一组试验按验证损失选出的完整 LoRA 模型：冻结基础权重（包含图像编码器和词嵌入）、非零适配器权重、完整架构构造参数、按 Token ID 排列的词表、LoRA 目标模块/rank/alpha/dropout、来源和目标描述顺序、预处理及训练配置、最佳轮次与验证损失。加载先构造基础架构，再注入 LoRA，最后严格加载全部权重；不需要重跑来源预训练。

旧 `trial_1_lora_adapter.pt` 是历史 adapter-only 文件，本身无法恢复对应的基础权重，新加载器会拒绝它。新训练入口不再生成该旧格式；既有文件不删除。当前仅保存第一组 LoRA 模型，未保存其他试验或完整微调对照模型。

模型输入为 RGB、float32、NCHW，像素除以 255，不做均值/标准差归一化。入口复用共享 BGR→RGB 转换；图片尺寸必须等于 checkpoint 的 `image_size`，不自动拉伸或裁剪。加载使用 `weights_only=True`、`strict=True`、`eval()`，模型和输入统一放到 `--device`（默认 CPU），无需读取训练配置文件。架构固定层结构由仓库代码及格式版本 1 定义。

已有**新格式 checkpoint** 后，从仓库根目录执行以下示例。它只生成一张合成输入图并加载推理，不训练；如果尚无新 checkpoint，不能从历史指标或旧 adapter 凭空恢复模型。

```bash
conda activate robot_learning
PYTHONPATH=src python - <<'PY'
import cv2
import torch
from robot_learning.vision_language import generate_image_text_dataset
checkpoint = torch.load("outputs/015_lora_language_adaptation/trial_1_lora_model.pt", map_location="cpu", weights_only=True)
size = checkpoint["preprocessing"]["image_size"]
image = generate_image_text_dataset(1, size, 123).images[0]
rgb = image.mul(255).round().byte().permute(1, 2, 0).numpy()
cv2.imwrite("/tmp/015_checkpoint_input.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
PY
PYTHONPATH=src python experiments/015_lora_language_adaptation/run.py --checkpoint outputs/015_lora_language_adaptation/trial_1_lora_model.pt --image /tmp/015_checkpoint_input.png --device cpu
```

标准输出为 JSON，包含目标别名顺序、余弦相似度和最佳匹配描述。替换 `--image` 可读取自己的同尺寸图片；合成任务模型的真实图像效果未验证。推理不会创建或改写训练输出。

验证命令：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_model_artifacts.py tests/test_object_detection.py tests/test_semantic_segmentation.py tests/test_peft.py`。新测试使用临时目录，包含权重/输出往返一致性、缺失/多余/形状错误权重的拒绝加载、三个独立进程推理入口，以及缩小数据规模的训练保存检查。微型测试通过不代表重新验证了历史训练分数。

## 当前边界

这是小型合成模型上的语言映射实验。LoRA 的计算、显存和存储优势在真实大模型上更明显；当前结果不能直接外推到大型 VLM 或 LLM。
