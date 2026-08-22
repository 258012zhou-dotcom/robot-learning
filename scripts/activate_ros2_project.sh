#!/usr/bin/env bash

# Activate ROS 2 only in terminals used for the learning workspace.
# This avoids adding ROS paths to every Conda/Python terminal.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "请使用 source scripts/activate_ros2_project.sh 激活当前终端。" >&2
    exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_setup="$project_root/projects/ros2_point_robot_ws/install/setup.bash"

source /opt/ros/humble/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=31
export ROS_LOCALHOST_ONLY=0

if [[ -f "$workspace_setup" ]]; then
    source "$workspace_setup"
else
    echo "提示：尚未找到工作空间 install/setup.bash，请先运行 colcon build。" >&2
fi

echo "ROS 2 项目环境已激活：RMW=$RMW_IMPLEMENTATION, DOMAIN=$ROS_DOMAIN_ID"
