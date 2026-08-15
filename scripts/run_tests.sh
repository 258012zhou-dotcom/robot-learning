#!/usr/bin/env bash

set -e

if [[ "${CONDA_DEFAULT_ENV:-}" != "robot_learning" ]]; then
  echo "请先运行：conda activate robot_learning"
  exit 1
fi

python -m pytest -q
