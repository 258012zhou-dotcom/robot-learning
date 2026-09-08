"""假总线测试，不导入 CAN 驱动，不连接机械臂。"""

from pathlib import Path
import runpy
from types import SimpleNamespace as NS

import pytest


MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1]
                           / "experiments/025_piper_safety_query/capture_firmware.py"))


def test_one_request_preserves_reply_bytes(monkeypatch):
    clock = iter([0, 0.1, 4])
    monkeypatch.setattr(MODULE["time"], "monotonic", lambda: next(clock))

    class Bus:
        sent = []
        closed = False
        def send(self, request, timeout):
            self.sent.append(request)
        def recv(self, timeout):
            return NS(arbitration_id=0x4AF, is_extended_id=False,
                      is_error_frame=False, is_remote_frame=False,
                      data=b'S-V1\x00\x00\x00\x00', dlc=8, timestamp=1.0)
        def shutdown(self):
            self.closed = True

    bus = Bus()
    request = object()
    frames = MODULE["capture"](bus, request)
    assert bus.sent == [request]
    assert frames[0]["hex"] == "53 2d 56 31 00 00 00 00"
    assert bus.closed
    assert MODULE["QUERY_ID"] == 0x4AF
    assert MODULE["QUERY_BYTES"] == bytes([1, 0, 0, 0, 0, 0, 0, 0])


def test_send_failure_closes_bus():
    class Bus:
        closed = False
        def send(self, request, timeout):
            raise RuntimeError("send failed")
        def shutdown(self):
            self.closed = True
    bus = Bus()
    with pytest.raises(RuntimeError, match="send failed"):
        MODULE["capture"](bus, object())
    assert bus.closed
