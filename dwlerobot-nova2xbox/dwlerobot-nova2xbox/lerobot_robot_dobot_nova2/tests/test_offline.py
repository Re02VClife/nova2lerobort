from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from lerobot_robot_dobot_nova2 import DobotNova2Config, DobotNova2Keyboard, DobotNova2KeyboardConfig
from lerobot_robot_dobot_nova2.dobot_sdk_adapter import DobotNova2OfficialSDK, clamp_delta_action, parse_dobot_reply


def test_parse_dobot_reply() -> None:
    reply = parse_dobot_reply("0,{250.1,-120.2,180,180,0,180},GetPose();")
    assert reply.ok
    assert reply.floats() == [250.1, -120.2, 180.0, 180.0, 0.0, 180.0]


def test_clamp_delta_action() -> None:
    clipped = clamp_delta_action({"delta_x": 5, "delta_rx": 3, "gripper": 2}, max_xyz_mm=1, max_rot_deg=0.5)
    assert clipped["delta_x"] == 1
    assert clipped["delta_rx"] == 0.5
    assert clipped["gripper"] == 1.0
    assert clipped["delta_y"] == 0.0


def test_configs_and_keyboard_features() -> None:
    robot_cfg = DobotNova2Config()
    keyboard = DobotNova2Keyboard(DobotNova2KeyboardConfig())
    assert robot_cfg.type == "dobot_nova2"
    assert keyboard.config.type == "dobot_nova2_keyboard"
    assert list(keyboard.action_features) == [
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_rx",
        "delta_ry",
        "delta_rz",
        "gripper",
    ]


class FakeDashboard:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, str]] = []

    def SetHoldRegs(self, index: int, addr: int, count: int, value: str, data_type: str) -> str:
        assert count == 1
        assert data_type == "U16"
        self.calls.append((index, addr, value))
        return "0,{},SetHoldRegs();"


def test_gripper_writes_skip_unchanged_cached_values() -> None:
    dashboard = FakeDashboard()
    sdk = DobotNova2OfficialSDK("127.0.0.1", 29999, 29999, Path("."))
    sdk.dashboard = dashboard
    sdk._gripper_index = 2

    with patch("lerobot_robot_dobot_nova2.dobot_sdk_adapter.time.sleep", lambda _seconds: None):
        sdk.set_gripper_percent(40000, 40008, 40, force_percent=30)
        sdk.set_gripper_percent(40000, 40008, 40, force_percent=30)
        sdk.set_gripper_percent(40000, 40008, 55, force_percent=30)

    assert dashboard.calls == [
        (2, 40090, "{1}"),
        (2, 40001, "{30}"),
        (2, 40000, "{40}"),
        (2, 40000, "{55}"),
    ]
