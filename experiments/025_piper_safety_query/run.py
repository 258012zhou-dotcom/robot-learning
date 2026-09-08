"""实验 025：默认预览；--query-once 才连接 CAN 并发送参数查询。"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from robot_learning.piper_safety_query import query_parameters


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-once", action="store_true")
    args = parser.parse_args()
    if not args.query_once:
        print("实验 025：查询固件、六关节角度/速度/加速度限制。")
        print("当前仅预览：未导入 Piper SDK，未连接 CAN。")
        return

    # 只使用已经检查过源码的 SDK 版本。
    sdk_version = version("piper_sdk")
    if sdk_version != "0.6.1":
        raise RuntimeError(f"SDK 版本为 {sdk_version}，需重新核对查询接口")
    from piper_sdk import C_PiperInterface

    result = query_parameters(C_PiperInterface(can_name="can0"))
    result.update(experiment_name="025_piper_safety_query", sdk_version=sdk_version)
    output = ROOT / "outputs/025_piper_safety_query/results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"保存到：{output}")


if __name__ == "__main__":
    main()
