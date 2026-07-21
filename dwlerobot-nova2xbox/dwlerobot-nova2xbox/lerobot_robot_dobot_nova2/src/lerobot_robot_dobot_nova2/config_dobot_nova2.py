from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

DEFAULT_DOBOT_SDK_PATH = Path(r"E:\jushenzhineng\DOBOT\TCP-IP-Python-V4-main\TCP-IP-Python-V4-main")


@RobotConfig.register_subclass("dobot_nova2")
@dataclass
class DobotNova2Config(RobotConfig):
    """DOBOT Nova2 机械臂配置。"""

    host: str = "192.168.5.1"
    dashboard_port: int = 29999
    move_port: int = 29999
    sdk_path: Path = DEFAULT_DOBOT_SDK_PATH
    connect_timeout_s: float = 3.0
    startup_on_connect: bool = False
    startup_power_on_wait_s: float = 10.0
    speed_factor: int = 5
    control_mode: str = "movl"
    max_xyz_step_mm: float = 1.0
    max_rot_step_deg: float = 0.5
    pose_limits: tuple[tuple[float, float], ...] | None = None
    enable_gripper: bool = False
    gripper_modbus_ip: str = "127.0.0.1"
    gripper_modbus_port: int = 60000
    gripper_slave_id: int = 1
    gripper_width_addr: int = 40000
    gripper_position_addr: int = 40005
    gripper_execute_addr: int = 40008
    gripper_find_travel_on_connect: bool = True
    gripper_find_travel_wait_s: float = 5.0
    gripper_max_width: int = 100
    gripper_step_percent: float = 5.0
    gripper_action_threshold: float = 0.5
    gripper_force_percent: int = 50
    gripper_execute_after_width: bool = False
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@TeleoperatorConfig.register_subclass("dobot_nova2_keyboard")
@dataclass
class DobotNova2KeyboardConfig(TeleoperatorConfig):
    """Nova2 键盘遥操器配置。"""

    xyz_step: float = 1.0
    rot_step: float = 0.5
    gripper_step: float = 1.0


@TeleoperatorConfig.register_subclass("dobot_nova2_xbox")
@dataclass
class DobotNova2XboxConfig(TeleoperatorConfig):
    """Nova2 Xbox 手柄遥操器配置。"""

    control_style: str = "smooth"
    xyz_step: float = 12.0
    rot_step: float = 5.0
    deadzone: float = 0.12
    axis_curve: float = 1.6
    smoothing: float = 0.35
    stop_smoothing: float = 0.65
    center_release_s: float = 0.12
    release_deadzone: float = 0.35
    release_drop: float = 0.2
    repeat_s: float = 0.15
    min_output: float = 0.25
    turbo_threshold: float = 0.85
    turbo_multiplier: float = 2.0
    trigger_threshold: float = 0.5
    left_x_axis: int = 0
    left_y_axis: int = 1
    right_x_axis: int = 2
    right_y_axis: int = 3
    lt_axis: int = 4
    rt_axis: int = 5
    lb_button: int = 4
    rb_button: int = 5
    a_button: int = 0
    b_button: int = 1
    x_button: int = 2
    y_button: int = 3
    invert_left_x: bool = True
    invert_left_y: bool = True
    invert_right_x: bool = True
    invert_right_y: bool = False
