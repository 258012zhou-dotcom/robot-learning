#!/usr/bin/env bash

set -e

if [[ "${CONDA_DEFAULT_ENV:-}" != "robot_learning" ]]; then
  echo "请先运行：conda activate robot_learning"
  exit 1
fi

# This project does not use ROS pytest plugins.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# WSL/Linux: 仿真库可能先加载系统旧版 libstdc++，使 Matplotlib 导入失败。
# 仅让本次测试优先使用当前 Python 环境的运行库，不修改系统或 ROS 配置。
test_runtime_lib="$(python -c 'import sys; print(sys.prefix + "/lib")')"
if [[ -f "$test_runtime_lib/libstdc++.so.6" ]]; then
  export LD_LIBRARY_PATH="$test_runtime_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

python -m pytest -q "$@"
