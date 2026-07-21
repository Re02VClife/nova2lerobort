from __future__ import annotations

import hashlib
import importlib.util
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


class DobotNova2SDKError(RuntimeError):
    """DOBOT Nova2 V4 SDK 适配层异常。"""


@dataclass(frozen=True)
class DobotReply:
    """DOBOT V4 SDK 返回字符串解析结果。"""

    code: int
    values: tuple[str, ...]
    command: str
    raw: str

    @property
    def ok(self) -> bool:
        return self.code == 0

    def floats(self) -> list[float]:
        return [float(v.strip()) for v in self.values if v.strip()]

    def ints(self) -> list[int]:
        return [int(float(v.strip())) for v in self.values if v.strip()]


class DobotNova2OfficialSDK:
    """通过 DOBOT TCP-IP Python V4 SDK 控制 Nova2。"""

    def __init__(
        self,
        host: str,
        dashboard_port: int,
        move_port: int,
        sdk_path: Path,
        speed_factor: int = 5,
        connect_timeout_s: float = 3.0,
    ) -> None:
        self.host = host
        self.dashboard_port = int(dashboard_port)
        # V4 的运动指令在 Dashboard 连接上发送；保留 move_port 参数仅用于 CLI 兼容和端口检测。
        self.move_port = int(move_port)
        self.sdk_path = Path(sdk_path)
        self.speed_factor = int(speed_factor)
        self.connect_timeout_s = float(connect_timeout_s)
        self.dashboard: Any | None = None
        self._sdk_module: Any | None = None
        self._gripper_index: int | None = None
        self._gripper_auto_disabled = False
        self._gripper_force_percent: int | None = None
        self._gripper_width_percent: int | None = None

    @property
    def is_connected(self) -> bool:
        return self.dashboard is not None

    @property
    def gripper_ready(self) -> bool:
        return self._gripper_index is not None

    def connect(self) -> None:
        if self.is_connected:
            return
        self._check_tcp_port()
        module = self._load_v4_sdk()
        self.dashboard = module.DobotApiDashboard(self.host, self.dashboard_port)
        self._silence_sdk_logger(self.dashboard)
        for method in ("MovL", "MovJ", "ServoP"):
            if not hasattr(self.dashboard, method):
                raise DobotNova2SDKError(
                    f"当前 SDK 的 DobotApiDashboard 缺少 {method}，请确认 sdk_path 指向 TCP-IP-Python-V4-main 内层目录。"
                )

    def _check_tcp_port(self) -> None:
        try:
            with socket.create_connection((self.host, self.dashboard_port), timeout=self.connect_timeout_s):
                return
        except OSError as exc:
            raise DobotNova2SDKError(
                f"无法连接 DOBOT dashboard {self.host}:{self.dashboard_port}，请检查网线、控制器 IP、电脑网卡 IP 和机器人上电状态。"
            ) from exc

    def startup(self, power_on_wait_s: float = 10.0) -> None:
        dashboard = self._require_dashboard()
        # 使能流程：请求控制权（如果控制器支持）-> 清错 -> 上电 -> 等待控制器完成上电 -> 使能 -> 设置速度比例。
        if hasattr(dashboard, "RequestControl"):
            try:
                self._checked(dashboard.RequestControl())
            except Exception:
                pass
        self._checked(dashboard.ClearError())
        self._checked(dashboard.PowerOn())
        time.sleep(max(0.0, float(power_on_wait_s)))
        self._checked(dashboard.EnableRobot())
        time.sleep(0.5)
        self._checked(dashboard.SpeedFactor(max(1, min(100, self.speed_factor))))

    def close(self) -> None:
        if self.dashboard is not None:
            try:
                self.dashboard.close()
            except Exception:
                pass
        self.dashboard = None
        self._gripper_index = None
        self._gripper_auto_disabled = False
        self._gripper_force_percent = None
        self._gripper_width_percent = None

    def robot_mode(self) -> int:
        reply = self._checked(self._require_dashboard().RobotMode())
        values = reply.ints()
        if not values:
            raise DobotNova2SDKError(f"RobotMode 没有返回模式值: {reply.raw}")
        return values[0]

    def get_joints(self) -> list[float]:
        reply = self._checked(self._require_dashboard().GetAngle())
        return self._expect_len(reply.floats(), 6, "GetAngle")

    def get_pose(self) -> list[float]:
        reply = self._checked(self._require_dashboard().GetPose())
        return self._expect_len(reply.floats(), 6, "GetPose")

    def move_delta(
        self,
        delta: Sequence[float],
        mode: str,
        pose_limits: Sequence[tuple[float, float]] | None = None,
        current_pose: Sequence[float] | None = None,
    ) -> list[float]:
        delta_values = self._expect_len([float(v) for v in delta], 6, "delta")
        current = self.get_pose() if current_pose is None else self._expect_len(
            [float(v) for v in current_pose], 6, "current_pose"
        )
        target = [current[i] + delta_values[i] for i in range(6)]
        if pose_limits is not None:
            target = self._clamp_pose(target, pose_limits)

        self._send_motion(target, mode)
        return target

    def emergency_stop(self) -> None:
        self._checked(self._require_dashboard().EmergencyStop())

    def init_gripper(
        self,
        ip: str,
        port: int,
        slave_id: int,
        force_percent: int = 50,
        find_travel_addr: int = 40008,
        find_travel_on_connect: bool = True,
        find_travel_wait_s: float = 5.0,
    ) -> int:
        dashboard = self._require_dashboard()
        for index in range(5):
            try:
                dashboard.sendRecvMsg(f"ModbusClose({index})")
            except Exception:
                pass
        # 与 Smartagent/QML 工程保持一致：先配置末端 485，再创建 RTU Modbus 通道。
        self._checked(dashboard.sendRecvMsg('SetTool485(115200,"N",1)'))
        time.sleep(0.5)
        reply = self._checked(dashboard.sendRecvMsg(f"ModbusCreate({ip},{int(port)},{int(slave_id)},1)"))
        values = reply.ints()
        if not values:
            raise DobotNova2SDKError(f"ModbusCreate 没有返回 index: {reply.raw}")
        self._gripper_index = values[0]
        time.sleep(0.3)
        self.gripper_set_force(force_percent)
        if find_travel_on_connect:
            self.gripper_find_travel(find_travel_addr, wait_s=find_travel_wait_s)
        return self._gripper_index

    def read_gripper_percent(self, position_addr: int, max_width: int = 100) -> float:
        if self._gripper_index is None:
            return float(max(1, int(max_width)))
        reply = self._checked(self._require_dashboard().GetHoldRegs(self._gripper_index, int(position_addr), 1, "U16"))
        values = reply.ints()
        width_max = max(1, int(max_width))
        return float(max(0, min(width_max, values[0]))) if values else float(width_max)

    def gripper_disable_auto_run(self) -> None:
        if self._gripper_index is None:
            return
        if self._gripper_auto_disabled:
            return
        self._checked(self._require_dashboard().SetHoldRegs(self._gripper_index, 40090, 1, "{1}", "U16"))
        self._gripper_auto_disabled = True

    def gripper_find_travel(self, find_travel_addr: int = 40008, wait_s: float = 5.0) -> None:
        if self._gripper_index is None:
            return
        self._checked(self._require_dashboard().SetHoldRegs(self._gripper_index, int(find_travel_addr), 1, "{1}", "U16"))
        time.sleep(max(0.0, float(wait_s)))

    def gripper_set_force(self, force_percent: int) -> None:
        if self._gripper_index is None:
            return
        value = max(0, min(100, int(round(force_percent))))
        if self._gripper_force_percent == value:
            return
        self._checked(self._require_dashboard().SetHoldRegs(self._gripper_index, 40001, 1, "{" + str(value) + "}", "U16"))
        self._gripper_force_percent = value

    def set_gripper_percent(
        self,
        width_addr: int,
        execute_addr: int,
        percent: float,
        force_percent: int = 50,
        execute_after_width: bool = False,
        max_width: int = 100,
    ) -> None:
        if self._gripper_index is None:
            return
        width_max = max(1, int(max_width))
        width_value = max(0, min(width_max, int(round(percent))))
        dashboard = self._require_dashboard()
        self.gripper_set_force(force_percent)
        self._checked(
            dashboard.SetHoldRegs(self._gripper_index, int(width_addr), 1, "{" + str(width_value) + "}", "U16")
        )
        self._gripper_width_percent = width_value
        # Smartagent 手动开/闭爪默认只写力度和开口，不自动写执行寄存器；保留开关兼容需要 Execute 的固件。
        if execute_after_width:
            time.sleep(0.25)
            self._checked(dashboard.SetHoldRegs(self._gripper_index, int(execute_addr), 1, "{1}", "U16"))
            time.sleep(0.1)
            self._checked(dashboard.SetHoldRegs(self._gripper_index, int(execute_addr), 1, "{0}", "U16"))

    def _send_motion(self, pose: Sequence[float], mode: str) -> DobotReply:
        dashboard = self._require_dashboard()
        normalized = mode.strip().lower()
        speed = max(1, min(100, self.speed_factor))
        if normalized == "movl":
            raw = dashboard.MovL(*pose, 0, a=speed, v=speed)
        elif normalized == "movj":
            raw = dashboard.MovJ(*pose, 0, a=speed, v=speed)
        elif normalized == "servop":
            raw = dashboard.ServoP(*pose)
        else:
            raise ValueError(f"不支持的 control_mode: {mode}")
        return self._checked(raw)

    def movej_joints(self, joints: Sequence[float]) -> DobotReply:
        """发送 MovJ 关节角指令。"""
        target_joints = [float(value) for value in joints]
        self._expect_len(target_joints, 6, "movej_joints")
        dashboard = self._require_dashboard()
        speed = max(1, min(100, self.speed_factor))
        raw = dashboard.MovJ(*target_joints, 1, a=speed, v=speed)
        return self._checked(raw)

    def _load_v4_sdk(self) -> Any:
        if self._sdk_module is not None:
            return self._sdk_module
        sdk_file = self.sdk_path / "dobot_api.py"
        if not sdk_file.is_file():
            raise FileNotFoundError(f"没有找到官方 V4 SDK 文件: {sdk_file}")
        digest = hashlib.md5(str(sdk_file).encode("utf-8")).hexdigest()  # nosec B324: 仅用于模块名去重。
        module_name = f"dobot_api_v4_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, sdk_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 DOBOT V4 SDK: {sdk_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self._sdk_module = module
        return module

    @staticmethod
    def _silence_sdk_logger(api: Any) -> None:
        # 官方 SDK 默认把每条 TCP 指令打印出来；采集数据时会很吵，这里改成静默。
        api.log = lambda _text: None

    def _require_dashboard(self) -> Any:
        if self.dashboard is None:
            raise DobotNova2SDKError("dashboard 尚未连接")
        return self.dashboard

    def _checked(self, raw: str) -> DobotReply:
        reply = parse_dobot_reply(raw)
        if not reply.ok:
            raise DobotNova2SDKError(f"DOBOT 指令失败: ErrorID={reply.code}, reply={raw}")
        return reply

    @staticmethod
    def _expect_len(values: list[float], expected_len: int, name: str) -> list[float]:
        if len(values) != expected_len:
            raise DobotNova2SDKError(f"{name} 需要 {expected_len} 个值，实际得到 {len(values)} 个: {values}")
        return values

    @staticmethod
    def _clamp_pose(pose: Sequence[float], limits: Sequence[tuple[float, float]]) -> list[float]:
        if len(limits) != 6:
            raise ValueError("pose_limits 需要 6 个 (min, max)")
        return [max(float(low), min(float(high), float(value))) for value, (low, high) in zip(pose, limits)]


def parse_dobot_reply(raw: str) -> DobotReply:
    text = raw.strip()
    if text.endswith(";"):
        text = text[:-1]
    first = text.split(",", 1)[0].strip()
    try:
        code = int(first)
    except ValueError as exc:
        raise DobotNova2SDKError(f"无法解析 DOBOT 回复: {raw}") from exc

    values: tuple[str, ...] = ()
    match = re.search(r"\{([^}]*)\}", text)
    if match:
        values = tuple(v.strip() for v in match.group(1).split(",") if v.strip())
    command = re.sub(r"^\s*-?\d+\s*,\s*\{[^}]*\}\s*,?", "", text, count=1).strip()
    return DobotReply(code=code, values=values, command=command, raw=raw)


def clamp_delta_action(action: dict[str, float], max_xyz_mm: float, max_rot_deg: float) -> dict[str, float]:
    """限制单步增量，避免策略或键盘误输入导致机械臂大幅跳动。"""
    limits = {
        "delta_x": float(max_xyz_mm),
        "delta_y": float(max_xyz_mm),
        "delta_z": float(max_xyz_mm),
        "delta_rx": float(max_rot_deg),
        "delta_ry": float(max_rot_deg),
        "delta_rz": float(max_rot_deg),
        "gripper": 1.0,
    }
    clipped: dict[str, float] = {}
    for key, limit in limits.items():
        value = float(action.get(key, 0.0))
        clipped[key] = max(-limit, min(limit, value))
    return clipped
