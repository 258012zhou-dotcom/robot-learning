# 阶段 0 检查点 1：开发环境与基础测试

## 已验证

- 项目路径：`~/AI_Project/robot-learning`
- 主学习环境：`conda activate robot_learning`
- Python：3.10.20
- 已确认 NumPy、Matplotlib 和 pytest 可用
- 基础测试命令：`python -m pytest -q`
- 第一个测试结果：2 passed
- 已创建一维机器人位置更新函数和正反方向测试

## 环境经验

ROS 2 自动加载会向 `PYTHONPATH` 加入 ROS 包路径，可能干扰普通 Conda Python 环境。

当前处理方式：

- 已注释 `~/.bashrc` 中自动加载 ROS 2 的命令；
- 普通 Python 学习时使用 `robot_learning`；
- 未来做 ROS 2 时，在专门的新终端手动执行：
  `source /opt/ros/humble/setup.bash`

## 当前未完成

- Linux 常用文件操作
- 依赖记录与可复现环境
- VS Code 开发流程
- pytest 的更多用法
- 日志、配置和随机种子管理
