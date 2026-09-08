"""首次使能反馈验证；默认预览，不是运动程序或安全停止器。"""

import argparse
from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
import time


FAULTS = ("voltage_too_low", "motor_overheating", "driver_overcurrent",
          "driver_overheating", "collision_status", "driver_error_status",
          "stall_status")


def snapshot(piper):
    """复制 SDK 缓存；这不是硬件同时采样或安全等级的反馈检查。"""
    arm = deepcopy(piper.GetArmStatus())
    drivers = deepcopy(piper.GetArmLowSpdInfoMsgs())
    motors = [getattr(drivers, f"motor_{i}") for i in range(1, 7)]
    return {
        "mode": int(arm.arm_status.ctrl_mode),
        "status": int(arm.arm_status.arm_status),
        "enabled": [bool(m.foc_status.driver_enable_status) for m in motors],
        "motor_ids": [int(m.can_id) for m in motors],
        "fault": any(getattr(m.foc_status, name) for m in motors for name in FAULTS),
        "arm_stamp": float(arm.time_stamp),
        "driver_stamp": float(drivers.time_stamp),
    }


def validate(sample):
    if sample["mode"] != 0 or sample["status"] != 0 or sample["fault"]:
        raise RuntimeError("模式不是待机或存在故障；不再发送命令")
    if len(set(sample["motor_ids"])) != 6 or 0 in sample["motor_ids"]:
        raise RuntimeError("未收到完整六轴反馈")
    if sample["arm_stamp"] <= 0 or sample["driver_stamp"] <= 0:
        raise RuntimeError("未收到有效状态反馈")


def observe_once(piper, report, *, clock=time.monotonic, sleep=time.sleep):
    """调用方先连接。失败后不重试，不自动失能或恢复；状态可能已改变。"""
    before = snapshot(piper)
    validate(before)
    if any(before["enabled"]):
        raise RuntimeError("初始必须六轴全部未使能；本次不会重复使能")
    sleep(0.15)
    fresh = snapshot(piper)
    validate(fresh)
    if (fresh["arm_stamp"] <= before["arm_stamp"]
            or fresh["driver_stamp"] <= before["driver_stamp"]
            or any(fresh["enabled"])):
        raise RuntimeError("反馈未更新或初始使能状态变化；取消实验")
    report["before"] = fresh
    # 标记尝试而非成功：发送抛错也可能已经到达设备。
    report["enable_attempted"] = True
    piper.EnableArm(7)
    deadline = clock() + 3.0
    last = fresh
    updated = [clock(), clock()]
    while clock() < deadline:
        current = snapshot(piper)
        report["last_feedback"] = current
        validate(current)
        for index, key in enumerate(("arm_stamp", "driver_stamp")):
            if current[key] > last[key]:
                updated[index] = clock()
            elif current[key] < last[key]:
                raise RuntimeError("反馈时间戳回退")
        if any(clock() - t > 0.5 for t in updated):
            raise RuntimeError("反馈更新中断；设备最终状态未知")
        if all(current["enabled"]):
            report["outcome"] = "all_enabled_reported"
            return
        last = current
        sleep(0.05)
    raise TimeoutError("3秒内未收到六轴全部使能；不重试，可能已有部分轴使能")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print("只尝试使能一次，无运动目标；不自动失能、复位或归位。")
    print("退出/Ctrl+C只结束本程序，不保证停机或保持；设备可能继续使能。")
    if not args.execute:
        print("预览结束：未导入SDK，未连接CAN。")
        return
    if version("piper_sdk") != "0.6.1":
        raise RuntimeError("SDK版本不符，请重新核对接口")
    answer = input("仅从臂连接、可靠支撑、人员离开运动范围、现场停止方案已就绪；输入 ENABLE 确认：")
    if answer != "ENABLE":
        print("取消，未连接CAN。")
        return
    from piper_sdk import C_PiperInterface

    report = {"enable_attempted": False, "outcome": "not_completed"}
    piper = C_PiperInterface(can_name="can0")
    exit_code = 0
    try:
        piper.ConnectPort(piper_init=False)
        time.sleep(1.0)
        observe_once(piper, report)
    except (Exception, KeyboardInterrupt) as error:
        exit_code = 1
        report["outcome"] = "interrupted_or_failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        # 仅释放程序连接，不代表停止机械臂。禁止在这里加恢复动作。
        try:
            piper.DisconnectPort()
        except Exception as error:
            exit_code = 1
            report["disconnect_error"] = str(error)
        output = Path(__file__).resolve().parents[2] / "outputs/025_piper_safety_query"
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"enable_observation_{time.time_ns()}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"记录：{path}")
        print("未发送停止/失能/归位命令。程序结束不等于机械臂安全停止。")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
