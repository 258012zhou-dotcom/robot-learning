"""记录固件查询原始帧；默认预览，--capture 才连接 CAN。"""

import argparse
import json
from pathlib import Path
import time


QUERY_ID = 0x4AF
QUERY_BYTES = bytes([1, 0, 0, 0, 0, 0, 0, 0])


def capture(bus, request, duration=3.0):
    """总线已创建并过滤；只发送一次请求，在固定窗口内接收。"""
    frames = []
    try:
        bus.send(request, timeout=0.5)
        deadline = time.monotonic() + duration
        while (remaining := deadline - time.monotonic()) > 0:
            message = bus.recv(timeout=remaining)
            if message is None:
                continue
            if (message.arbitration_id != QUERY_ID or message.is_extended_id
                    or message.is_error_frame or message.is_remote_frame):
                continue
            payload = bytes(message.data)
            frames.append({
                "timestamp": message.timestamp,
                "dlc": message.dlc,
                "hex": payload.hex(" "),
                # repr 会显式保留 \x00；hex 才是完整字节证据。
                "ascii_repr": repr(payload),
            })
    finally:
        bus.shutdown()
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if not args.capture:
        print("预览：一次标准帧 0x4AF 查询，数据 01 00 00 00 00 00 00 00；接收 3 秒。")
        print("当前未连接 CAN。")
        return
    import can

    request = can.Message(arbitration_id=QUERY_ID, data=QUERY_BYTES,
                          is_extended_id=False, check=True)
    bus = can.Bus(interface="socketcan", channel="can0", receive_own_messages=False,
                  can_filters=[{"can_id": QUERY_ID, "can_mask": 0x7FF, "extended": False}])
    frames = capture(bus, request)
    result = {"request_hex": QUERY_BYTES.hex(" "), "can_id": "0x4AF",
              "frames": frames, "frame_count": len(frames)}
    directory = Path(__file__).resolve().parents[2] / "outputs/025_piper_safety_query"
    directory.mkdir(parents=True, exist_ok=True)
    # 唯一文件名保留多次抓取，避免覆盖此前证据。
    path = directory / f"firmware_frames_{time.time_ns()}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"收到 {len(frames)} 帧。原始数据保存在：{path}")
    print("原始帧可能包含设备标识，请先本地查看，不直接公开全部数据。")
    if not frames:
        raise SystemExit("没有回复；不能据此判断固件版本。")


if __name__ == "__main__":
    main()
