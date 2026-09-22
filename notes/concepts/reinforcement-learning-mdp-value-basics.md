# 强化学习：MDP、回报与价值函数基础

## 快速复习

- 强化学习通过环境奖励学习策略，不需要每个状态都有专家动作标签。
- MDP 写作 `(S, A, P, R, γ)`：状态、动作、转移、奖励和折扣因子。
- 马尔可夫性质要求当前状态包含预测下一状态分布所需的信息，不要求环境一定是确定性的。
- Reward 是一步反馈；Return 是从当前时刻开始的累计折扣奖励。
- Policy 决定怎样选动作；`V` 评价状态；`Q` 评价状态下的指定动作；Advantage 表示动作比当前策略平均水平好多少。
- 一次轨迹 Return 只是随机样本，不等于价值期望。Monte Carlo 用多次完整轨迹的样本均值估计价值。

## 1. 强化学习的交互过程

Behavior Cloning 使用专家动作作为监督标签；强化学习通常只从环境获得状态变化和奖励：

```text
状态 s_t
→ 策略选择动作 a_t
→ 环境执行动作
→ 返回奖励 r_t 和下一状态 s_(t+1)
→ 重复
```

智能体的目标不是让某一步 Reward 最大，而是让长期 Return 的期望最大。

## 2. MDP 的组成

```text
MDP = (S, A, P, R, γ)
```

- `S`，State：环境当前处境。
- `A`，Action：智能体可以执行的动作。
- `P`，Transition：当前状态和动作怎样决定下一状态的概率分布。
- `R`，Reward：一步转移产生的反馈。
- `γ`，Discount Factor：未来奖励在当前回报中的权重，范围为 `[0, 1]`。

策略通常写作 `π(a|s)`，表示状态 `s` 下选择动作 `a` 的方式。它可以是确定性的，也可以是一个动作概率分布。

### 马尔可夫性质

给定当前状态和动作后，下一状态的概率分布不再依赖更早的历史。当前状态必须包含足够的信息。

点机器人只有位置时不满足这个要求：相同位置、不同速度的机器人执行同一动作后会到达不同位置。加入速度后，状态才能区分这两种运动趋势。

策略实际接收的是 Observation，不一定等于环境完整 State。当前一维任务的观察包含位置、速度、目标和目标误差，可以先近似按完全可观测 MDP 理解；相机遮挡、动作延迟和历史依赖可能形成 POMDP。

## 3. Reward、Return 与折扣因子

从时刻 `t` 开始的折扣回报为：

```text
G_t = r_t + γr_(t+1) + γ²r_(t+2) + ...
```

也可以从 Episode 末尾向前递推：

```text
G_t = r_t + γG_(t+1)
```

- `γ=0`：只计算当前一步奖励。
- `γ` 接近 1：更多考虑长期结果。
- `γ=1`：有限 Episode 中，Return 等于未折扣累计奖励。

`γ` 会改变优化目标，不只是改变日志显示。它还需要和控制周期一起理解：有效步数可以粗略看作 `1 / (1 - γ)`，有效物理时间还要乘以每个控制步的时间。

实验 030 比较了立即奖励 `[1]` 与延迟奖励 `[0, 2]`：

```text
G_quick = 1
G_patient = 2γ
```

所以 `γ<0.5` 时立即奖励更高，`γ=0.5` 时打平，`γ>0.5` 时延迟奖励更高。

当前点机器人的 Reward 是负距离减去动作成本。更接近 0 的奖励通常更好，但不同 `γ` 定义了不同目标，不能直接跨 `γ` 比较回报数值来判断策略优劣。

## 4. Policy、V、Q 与 Advantage

### Policy

```text
π(a|s)
```

描述状态 `s` 下怎样选择动作 `a`。

### 状态价值 V

```text
V^π(s)
```

表示从状态 `s` 开始，之后一直按照策略 `π` 行动时，未来 Return 的期望。

### 动作价值 Q

```text
Q^π(s,a)
```

表示在状态 `s` 先执行指定动作 `a`，之后再按照策略 `π` 行动时，未来 Return 的期望。

### Advantage

```text
A^π(s,a) = Q^π(s,a) - V^π(s)
```

- `A>0`：动作比当前策略在该状态下的平均水平更好。
- `A<0`：动作比平均水平更差。
- `A≈0`：动作没有明显优劣。

离散随机策略下：

```text
V^π(s) = Σ π(a|s)Q^π(s,a)
Σ π(a|s)A^π(s,a) = 0
```

因此 Advantage 衡量的是相对当前策略的改进方向，而不是绝对回报是否为正。

## 5. Return 样本与 Value 期望

同一个状态多次运行，可能得到不同 Return。一次 `G_t` 只是某一种未来结果；`V` 和 `Q` 是这些可能结果的期望。

Monte Carlo 方法等待 Episode 完成后计算实际 Return，再通过样本平均估计价值：

```text
V(s) ≈ 从状态 s 出发的多个 Return 平均值
Q(s,a) ≈ 在状态 s 先执行 a 后的多个 Return 平均值
```

实验 031 的真实答案为：

```text
Q_safe = 1.0
Q_risky = 1.8
V = 1.2
A_safe = -0.2
A_risky = 0.6
```

但只有 10 次采样时，`Q_risky` 被估计为 0，使真实为正的 Advantage 被错误估计为负。10000 次采样时，`Q_risky` 估计为 1.835，才接近真实值 1.8。

这说明少量样本不仅会带来数值误差，还可能让策略更新方向出错。增加样本通常能降低误差，但随机波动不保证误差随每一次扩充都严格单调下降。

## 6. TD 与 Bootstrapping

Monte Carlo 等待完整 Episode 后使用实际 Return。一步 TD 使用：

```text
TD target = r_t + γV(s_(t+1))
TD error  = TD target - V(s_t)
V(s_t) ← V(s_t) + α × TD error
```

其中 `α` 是学习率。使用下一状态的当前估计来更新当前状态，叫作 Bootstrapping。

如果转移真正终止，未来价值为 0：

```text
TD target = r_t                  if terminated
TD target = r_t + γV(s_(t+1))   otherwise
```

`truncated` 可能只是时间限制，不一定表示任务自然结束，是否继续 Bootstrapping 要根据任务定义处理。

实验 032 的两状态链中，TD 先从终止奖励学到 `V(S1)`，再在后续 Episode 中把价值逐渐传播到 `V(S0)`。Monte Carlo 在这个确定性短链中收敛更快，因为完整 Return 没有随机方差；这不是一般优劣结论。

## 7. 当前已验证与尚未验证

已经实际验证：

- 有限 Episode 的折扣回报及其向后递推。
- `γ` 对立即奖励和延迟奖励权衡的影响。
- `V = ΣπQ`、`A = Q - V` 和策略加权 Advantage 为 0。
- Monte Carlo 状态价值和动作价值估计。
- 少量随机样本可能误判 Advantage 的正负号。
- 一步 TD target、TD Error、学习率更新和终止状态处理。
- Bootstrapping 将下一状态估计逐步向前传播。

尚未验证：

- 使用神经网络拟合价值函数。
- Q-Learning、Policy Gradient、GAE 和 PPO。
- 点机器人上的强化学习策略训练。

## 8. 下一步

下一知识节点是 Q-Learning：把 TD target 从状态价值扩展到动作价值，并区分当前行为策略与目标中的贪心动作。在此基础上再进入 Policy Gradient 和 PPO。

## 项目入口

- [实验 030：MDP 与折扣回报](../../experiments/030_mdp_return/README.md)
- [实验 031：V、Q、Advantage 与 Monte Carlo 估计](../../experiments/031_value_estimation/README.md)
- [实验 032：TD 与 Bootstrapping](../../experiments/032_td_bootstrapping/README.md)
- [阶段路线](../../roadmap.md)
