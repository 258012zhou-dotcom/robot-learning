# ROS 2 工作空间与功能包

## 快速复习

- 工作空间（workspace）组织一个或多个功能包及其构建结果。
- 功能包（package）是 ROS 2 代码、依赖、资源和运行入口的基本单位。
- `package.xml` 描述 ROS 元数据和依赖；`setup.py` 描述 Python 安装规则。
- `colcon build` 生成构建结果；`source install/setup.bash` 将结果叠加到当前终端。
- 构建成功、规范测试通过和功能运行正确是三个不同层次的验证。

## 工作空间结构

```text
workspace/
├── src/       # 源码
├── build/     # 中间文件
├── install/   # 安装结果与环境脚本
└── log/       # 构建和测试日志
```

项目只提交 `src/` 和说明文档。其余目录由 colcon 生成，可以重新构建。

## Python 功能包的关键文件

- `package.xml`：包名、版本、维护者、许可证、运行依赖、测试依赖和构建类型。
- `setup.py`：安装 Python 模块、ament 索引资源、`package.xml` 和节点命令入口。
- `setup.cfg`：把节点脚本安装到 `lib/<package_name>`，供 `ros2 run` 查找。
- `resource/<package_name>`：将包登记到 ament 资源索引。
- `<package_name>/__init__.py`：Python 模块入口。

外层 `point_robot_ros/` 是 ROS 功能包根目录，内层同名目录是可以被 Python 导入的模块。

## 环境叠加

```text
/opt/ros/humble/setup.bash
  → workspace/install/setup.bash
  → 当前终端能发现系统包和本地包
```

环境变量属于当前进程。构建后不执行第二次 `source`，ROS 就可能找不到新包；删除工作空间也不会自动清除旧终端中的路径。

本项目根目录的机器学习代码使用 Conda，ROS 2 Humble 使用 `/usr/bin/python3`。两个环境使用独立终端，避免 Python 包和 pytest 插件冲突。

## 构建与测试

`colcon build --symlink-install` 扫描 `src/` 并构建包。符号链接安装便于开发 Python 源码，但修改包元数据、资源或命令入口后仍应重新构建。

`colcon test` 运行包注册的测试，`colcon test-result --verbose` 汇总结果。当前三个模板测试分别检查：

- Flake8：Python 代码风格。
- PEP 257：文档字符串格式。
- Copyright：版权头，当前主动跳过。

测试中的 `assert rc == 0` 要求检查工具返回成功状态。规范测试通过不能证明 ROS 节点已经通信，后续仍需单元测试和实际运行验证。

## 当前项目证据

- 用户亲自创建了工作空间和 `ament_python` 功能包。
- `point_robot_ros` 构建成功，并能由 `ros2 pkg prefix` 发现。
- 包使用独立 `pytest.ini`，避免读取根项目的 pytest 配置。
- 实际测试结果为 3 tests、0 errors、0 failures、1 skipped。

## 易错点

- 在 Conda 环境中运行 ROS 2 Python 包。
- 构建后忘记加载 `install/setup.bash`。
- 把生成的 `build/`、`install/` 和 `log/` 提交 Git。
- 修改 `setup.py` 中的节点入口后忘记重新构建。
- 看到发布日志或规范测试通过，就误认为进程间通信已经成功。
