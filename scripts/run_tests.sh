#!/usr/bin/env bash

set -e

if [[ "${CONDA_DEFAULT_ENV:-}" != "robot_learning" ]]; then
  echo "请先运行：conda activate robot_learning"
  exit 1
fi

# This project does not use ROS pytest plugins.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

python -m pytest -q
