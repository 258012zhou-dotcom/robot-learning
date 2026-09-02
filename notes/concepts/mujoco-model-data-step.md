# MuJoCo 模型、状态与物理步进

## 快速复习

- MJCF XML 描述机器人结构、关节、几何、质量、执行器和仿真参数。
- `MjModel` 保存编译后且运行中基本不变的模型；`MjData` 保存时间、状态、控制和动力学中间量。
- `qpos` 是广义位置，`qvel` 是广义速度，`ctrl` 是执行器输入。
- 最小循环是“写入 `ctrl` → 调用 `mj_step` → 读取 `qpos/qvel`”。
- 控制输入不是目标位置。状态变化取决于质量、阻尼、重力、接触、约束和积分器。
- `nq` 不一定等于 `nv`；读取复杂关节时应使用模型提供的地址，而不是猜数组索引。

## MJCF 的基本结构

```text
<mujoco>
├── <option>       时间步长、重力和积分器
├── <worldbody>    世界中的刚体与几何
│   └── <body>
│       ├── <joint> 允许的自由度
│       └── <geom>  形状、质量、碰撞和外观
└── <actuator>     电机或其他执行器
```

`body` 表示刚体坐标系，`joint` 决定子刚体相对父级可以怎样运动，`geom` 提供形状及接触/惯性相关信息，`actuator` 把控制输入连接到关节。

## MjModel 与 MjData

```python
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
```

`model` 包含：

- `model.nq`：广义位置维度。
- `model.nv`：广义速度维度。
- `model.nu`：控制输入维度。
- `model.opt.timestep`：一次 `mj_step` 推进的仿真时间。
- 关节、执行器、质量、阻尼及约束参数。

`data` 包含：

- `data.time`：当前仿真时间。
- `data.qpos`：广义位置。
- `data.qvel`：广义速度。
- `data.ctrl`：执行器控制输入。
- `data.qacc`：计算得到的广义加速度。

`MjData` 是状态容器。对同一个 `MjModel` 新建两个 `MjData`，可以运行两条互不影响的轨迹。

## 名称和状态地址

工程中应通过 MJCF 名称查找对象：

```python
joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "x_slide",
)
qpos_address = model.jnt_qposadr[joint_id]
qvel_address = model.jnt_dofadr[joint_id]
```

不要默认“第 i 个关节的位置一定在 `qpos[i]`”。自由关节和球关节使用四元数等表示，可能导致 `nq != nv`。

## 物理步进

```python
data.ctrl[actuator_id] = action
mujoco.mj_step(model, data)

position = data.qpos[qpos_address]
velocity = data.qvel[qvel_address]
```

若时间步长是 `0.01 s`，执行 300 次 `mj_step`，理论仿真时间为 `3.0 s`。实际浮点数可能显示为 `2.99999999999998`，应使用容差比较。

## 实验 017 的动力学证据

单滑动关节点机器人参数为：质量 `m=1`、阻尼 `b=0.4`、恒定控制 `u=1`、零重力。近似动力学为：

\[
m\dot v=u-bv
\]

稳态速度为：

\[
v_\infty=\frac{u}{b}=2.5\ \mathrm{m/s}
\]

三秒时理论速度约为 `1.7470 m/s`。实际 MuJoCo 结果：

- 最终位置：`3.1325 m`。
- 最终速度：`1.7470 m/s`。
- 模型维度：`nq=1`、`nv=1`、`nu=1`。
- 6 个模型与仿真单元测试通过。

初次模型的关节范围为 `[-2, 2]`，持续控制使机器人碰到上限并停住。放宽到 `[-5, 5]` 后才观察到纯粹的阻尼动力学。这说明关节限位也是动力学结果的一部分，实验设计必须隔离变量。

## 当前边界

- 实验是单自由度、零重力、无复杂接触的确定性仿真。
- MuJoCo 验证不等于真实机器人验证。
- 后续实验 018 已在该模型上定义 Observation、Action、Reward、Episode 和 Gymnasium 环境；详见[Gymnasium 环境、Episode 与策略评估](gymnasium-environment-and-rollout.md)。
- WSLg 中已验证 passive Viewer 可以正常显示该模型。Viewer 只负责显示状态，不应取代数值测试和结果记录。

## 仿真时间与显示时间

```python
mujoco.mj_step(model, data)  # 推进 model.opt.timestep
viewer.sync()                # 把当前 data 显示到窗口
time.sleep(...)              # 控制墙钟播放速度
```

删除 `sleep` 会让程序更快完成，但只要执行相同数量的 `mj_step`，仿真时间和理想确定性轨迹不变。渲染帧率、墙钟时间和物理仿真步长是三个不同概念。
