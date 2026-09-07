"""读取一次 Piper 机械臂反馈；默认只检查配置。"""

import argparse
from importlib.metadata import version
import json
import logging
from pathlib import Path
import time
from typing import Any

from robot_learning.piper_telemetry import (
    PIPER_JOINT_COUNT,
    gripper_raw_to_meters,
    joint_raw_to_degrees,
    joint_raw_to_radians,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "024_arm_inventory.json"
RESULT_PATH = PROJECT_ROOT / "outputs" / "024_arm_inventory" / "results.json"


def load_config() -> dict[str, Any]:
    """读取并检查实验配置。"""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    required = {
        "experiment_name",
        "hardware_model",
        "can_name",
        "conda_environment",
        "expected_sdk_version",
        "warmup_seconds",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing configuration fields: {missing}")
    if float(config["warmup_seconds"]) <= 0:
        raise ValueError("warmup_seconds must be positive")
    return config


def read_once(config: dict[str, Any]) -> dict[str, object]:
    """连接 CAN，读取一次反馈，然后立即断开。"""
    if str(config["hardware_model"]).startswith("TO_BE_CONFIRMED"):
        raise RuntimeError("请先根据机械臂铭牌填写 hardware_model")

    try:
        from piper_sdk import C_PiperInterface
    except ImportError as error:
        raise RuntimeError(
            "当前环境没有 piper_sdk，请使用已安装该 SDK 的 xvla 环境"
        ) from error

    piper = C_PiperInterface(can_name=str(config["can_name"]))

    # piper_init=False：只启动 CAN 接收线程，不执行 SDK 初始化查询。
    piper.ConnectPort(piper_init=False)
    try:
        time.sleep(float(config["warmup_seconds"]))

        joint_feedback = piper.GetArmJointMsgs()
        gripper_feedback = piper.GetArmGripperMsgs()
        arm_feedback = piper.GetArmStatus()

        joint_message = joint_feedback.joint_state
        joint_raw = tuple(
            int(getattr(joint_message, f"joint_{index}"))
            for index in range(1, PIPER_JOINT_COUNT + 1)
        )
        gripper_message = gripper_feedback.gripper_state
        arm_status = arm_feedback.arm_status
        gripper_status = gripper_message.foc_status

        result = {
            "experiment_name": config["experiment_name"],
            "hardware_model": config["hardware_model"],
            "can_name": config["can_name"],
            "piper_sdk_version": version("piper_sdk"),
            "joint_feedback_hz": float(joint_feedback.Hz),
            "joint_positions_raw": list(joint_raw),
            "joint_positions_deg": list(joint_raw_to_degrees(joint_raw)),
            "joint_positions_rad": list(joint_raw_to_radians(joint_raw)),
            "gripper_feedback_hz": float(gripper_feedback.Hz),
            "gripper_position_raw": int(gripper_message.grippers_angle),
            "gripper_position_m": gripper_raw_to_meters(
                gripper_message.grippers_angle
            ),
            "gripper_status": {
                "driver_enabled": bool(gripper_status.driver_enable_status),
                "homed": bool(gripper_status.homing_status),
                "sensor_error": bool(gripper_status.sensor_status),
                "driver_error": bool(gripper_status.driver_error_status),
            },
            "motor_enabled": list(piper.GetArmEnableStatus()),
            "arm_status": {
                "feedback_hz": float(arm_feedback.Hz),
                "control_mode": int(arm_status.ctrl_mode),
                "status_code": int(arm_status.arm_status),
                "motion_status": int(arm_status.motion_status),
            },
            "safety": {
                "piper_init": False,
                "enable_command_sent": False,
                "motion_command_sent": False,
            },
        }
    finally:
        # 这里只关闭 SDK 接收和 CAN 连接，不调用 ACT 的运动清理函数。
        piper.DisconnectPort()

    return result


def save_result(result: dict[str, object]) -> None:
    """将一次反馈快照保存到不提交 Git 的 outputs 目录。"""
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--read-once",
        action="store_true",
        help="connect to can0 and read one feedback snapshot",
    )
    return parser.parse_args()


def main() -> None:
    """默认预览配置；显式 --read-once 才连接 CAN。"""
    args = parse_args()
    config = load_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.read_once:
        logging.info("配置有效：%s", CONFIG_PATH)
        logging.info("机械臂型号：%s", config["hardware_model"])
        logging.info("使用环境：%s", config["conda_environment"])
        logging.info("当前未连接 CAN，也没有发送任何命令")
        return

    result = read_once(config)
    save_result(result)
    logging.info("关节角（度）：%s", result["joint_positions_deg"])
    logging.info("夹爪相对行程（米）：%.4f", result["gripper_position_m"])
    logging.info("夹爪状态：%s", result["gripper_status"])
    logging.info("机械臂状态码：%s", result["arm_status"]["status_code"])
    logging.info("结果文件：%s", RESULT_PATH)


if __name__ == "__main__":
    main()
