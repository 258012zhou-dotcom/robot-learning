# 实验 052：低维 ACT 的动作延迟边界

## 目标与原理

实验 051 的 40/40 来自无部署偏差的一维仿真。本次冻结同一低维 ACT checkpoint，只改变环境的**动作延迟**：`0、3、10` 个控制步，分别代表原协议、项目实验 022 用过的延迟量，以及一档更强的压力检查。环境对每条 Episode 独立清空延迟队列，延迟 `N` 表示前 `N` 步执行零动作；不改变传感器观察、动作增益、目标或模型参数。

每档都用同一组 seed 1000–1039，对比 `z=0` 下“最新第一动作”和“时间集成”两种执行方式。这是**预先固定的单变量仿真压力检查**，不是在评价集上搜索最佳超参数，更不是 ACT 论文复现。

## 验证方式

复用环境已有的动作延迟单元测试及 ACT 推理策略测试。实际运行核对来源数据与 checkpoint、seed 不与示范重叠、所有条件的初始目标相同；无延迟档须逐 Episode 复现实验 051。记录每条 Episode 的成功、步数、物理终点距离（本次观察没有偏差）、终点速度、超调和回报；失败 Episode 的观察/动作序列单独压缩保存。

```bash
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python -m pytest -q tests/test_act_lowdim_policy.py tests/test_point_robot_reach_env.py
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/052_act_action_delay_boundary/run.py
PYTHONPATH=src /home/zxd/miniconda3/envs/robot_learning/bin/python experiments/052_act_action_delay_boundary/diagnose_expert.py
```

配置见 `configs/052_act_action_delay_boundary.json`；输出在 Git 忽略的 `outputs/052_act_action_delay_boundary/`。需要实验 020 数据、实验 047 checkpoint 和实验 051 的参考结果。第三条命令是看到 ACT 延迟 10 步失败后才补做的**事后诊断**，不是预先设定的 ACT 主对照。

## 实际结果与边界

已执行上述两份相关测试：`68 passed, 2 warnings`（Gymnasium 无界 Box 的警告）。ACT 主实验运行成功，延迟 0 的逐 Episode 结果与实验 051 相符，40 个评价 seed 与示范数据不重叠，各条件初始目标相同。

| 动作延迟 | 最新第一动作 | 时间集成 | 平均完成步数：最新 / 集成 | 平均终点距离：最新 / 集成 |
| --- | --- | --- | --- | --- |
| 0 步 | 40/40 | 40/40 | 224.3 / 227.8 | 0.0400 / 0.0397 |
| 3 步 | 40/40 | 40/40 | 277.3 / 271.2 | 0.0354 / 0.0351 |
| 10 步 | 0/40 | 0/40 | 400.0 / 400.0 | 0.2451 / 0.2636 |

延迟 3 步保持成功，但比原协议平均多约 47–53 步；只看成功率会漏掉这一退化。延迟 10 步时两种方式均触及 400 步时限，80 条失败轨迹的观察/动作序列保存在 `failure_trajectories.npz`。例如 seed 1000 的最新第一动作在第 400 步距离目标仍有 0.2673，速度为 0.1938，不能算作成功。

事后用原始示教的比例控制器（P expert）在同一组目标、seed、延迟档上诊断：0 步 40/40，3 步 40/40，10 步也为 0/40。结果单独保存在 `expert_diagnostic.json`，不并入预先固定的 ACT 主对照。这提示 10 步延迟可能触及原控制器和环境闭环的共同边界；**不能据此认定是 ACT 特有缺陷，也不能认定时间集成能解决大延迟**。

这里只改变动作执行延迟，没有改变观测、任务复杂度或示范数据。结论只适用于这套一维仿真、40 个 seed 与 400 步时限；不是完整 ACT 论文复现，也不外推真实机械臂。
