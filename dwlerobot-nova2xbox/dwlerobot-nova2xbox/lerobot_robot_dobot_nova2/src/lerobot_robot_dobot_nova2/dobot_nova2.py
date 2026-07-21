from __future__ import annotations

import logging
import threading
import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_dobot_nova2 import DobotNova2Config
from .dobot_sdk_adapter import DobotNova2OfficialSDK, clamp_delta_action

logger = logging.getLogger(__name__)

INITIAL_MOVJ_JOINTS = [70.0, 30.0, 90.0, -30.0, -90.0, -10.0]


class DobotNova2(Robot):
    """LeRobot Robot：通过 DOBOT 官方 TCP-IP Python SDK 控制 Nova2。"""

    config_class = DobotNova2Config
    name = "dobot_nova2"

    def __init__(self, config: DobotNova2Config):
        super().__init__(config)
        self.config = config
        self.sdk = DobotNova2OfficialSDK(
            host=config.host,
            dashboard_port=config.dashboard_port,
            move_port=config.move_port,
            sdk_path=config.sdk_path,
            speed_factor=config.speed_factor,
            connect_timeout_s=config.connect_timeout_s,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self._gripper_percent = 100.0
        self._last_target_pose: list[float] | None = None
        self._last_observed_pose: list[float] | None = None
        self._sdk_lock = threading.RLock()

    @cached_property
    def observation_features(self) -> dict:
        features: dict[str, type | tuple[int, int, int]] = {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,
            "tcp.x": float,
            "tcp.y": float,
            "tcp.z": float,
            "tcp.rx": float,
            "tcp.ry": float,
            "tcp.rz": float,
            "gripper.pos": float,
        }
        for name, cam in self.cameras.items():
            features[name] = (self.config.cameras[name].height, self.config.cameras[name].width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_rx": float,
            "delta_ry": float,
            "delta_rz": float,
            "gripper": float,
        }

    @property
    def is_connected(self) -> bool:
        return self.sdk.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.sdk.connect()
        if self.config.startup_on_connect:
            self.sdk.startup(power_on_wait_s=self.config.startup_power_on_wait_s)
        if self.config.enable_gripper:
            self.sdk.init_gripper(
                self.config.gripper_modbus_ip,
                self.config.gripper_modbus_port,
                self.config.gripper_slave_id,
                force_percent=self.config.gripper_force_percent,
                find_travel_addr=self.config.gripper_execute_addr,
                find_travel_on_connect=self.config.gripper_find_travel_on_connect,
                find_travel_wait_s=self.config.gripper_find_travel_wait_s,
            )
            self._gripper_percent = self.sdk.read_gripper_percent(self.config.gripper_position_addr, self.config.gripper_max_width)
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        # Nova2 本体标定由 DOBOT 控制器和 DobotStudio 管理，LeRobot 侧不写电机标定。
        return None

    def configure(self) -> None:
        # 当前只设置全局速度比例；工具坐标系、负载、碰撞等级建议现场确认后再扩展。
        if self.sdk.dashboard is not None:
            self.sdk.dashboard.SpeedFactor(max(1, min(100, int(self.config.speed_factor))))

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        # SDK 的 dashboard TCP 连接是同步收发，避免录制线程和控制线程同时读写同一 socket。
        with self._sdk_lock:
            joints = self.sdk.get_joints()
            pose = self.sdk.get_pose()
        self._last_observed_pose = pose
        obs: dict[str, Any] = {
            "joint_1.pos": joints[0],
            "joint_2.pos": joints[1],
            "joint_3.pos": joints[2],
            "joint_4.pos": joints[3],
            "joint_5.pos": joints[4],
            "joint_6.pos": joints[5],
            "tcp.x": pose[0],
            "tcp.y": pose[1],
            "tcp.z": pose[2],
            "tcp.rx": pose[3],
            "tcp.ry": pose[4],
            "tcp.rz": pose[5],
            "gripper.pos": self._gripper_percent,
        }
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.read_latest()
        logger.debug("%s read observation: %.1fms", self, (time.perf_counter() - start) * 1e3)
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        clipped = clamp_delta_action(action, self.config.max_xyz_step_mm, self.config.max_rot_step_deg)
        motion_delta = [
            clipped["delta_x"],
            clipped["delta_y"],
            clipped["delta_z"],
            clipped["delta_rx"],
            clipped["delta_ry"],
            clipped["delta_rz"],
        ]
        if any(abs(value) > 1e-9 for value in motion_delta):
            # 高频遥操线程可能比 observation 更新更快，优先基于上次目标位姿继续累积小步。
            current_pose = self._last_target_pose if self._last_target_pose is not None else self._last_observed_pose
            with self._sdk_lock:
                self._last_target_pose = self.sdk.move_delta(
                    motion_delta,
                    mode=self.config.control_mode,
                    pose_limits=self.config.pose_limits,
                    current_pose=current_pose,
                )
        else:
            self._last_target_pose = self._last_observed_pose
        sent_action = dict(clipped)
        sent_action["gripper"] = self._write_gripper(clipped["gripper"])
        return sent_action

    @check_if_not_connected
    def move_to_initial_position(self) -> None:
        # 手柄 Y 键触发的固定初始位，等价于 robotControl.movJ(70, 30, 90, -30, -90, -10)。
        with self._sdk_lock:
            self.sdk.movej_joints(INITIAL_MOVJ_JOINTS)
            self._last_target_pose = None
            self._last_observed_pose = None

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()
        self.sdk.close()
        logger.info("%s disconnected.", self)

    def _read_gripper(self) -> float:
        if self.config.enable_gripper and self.sdk.gripper_ready:
            self._gripper_percent = self.sdk.read_gripper_percent(self.config.gripper_position_addr, self.config.gripper_max_width)
        return self._gripper_percent

    def _write_gripper(self, delta: float) -> float:
        # 策略输出是连续值，小幅噪声不应触发夹爪开合。
        threshold = max(0.0, float(self.config.gripper_action_threshold))
        if not self.config.enable_gripper or not self.sdk.gripper_ready or abs(delta) < threshold:
            return 0.0
        command = 1.0 if delta > 0 else -1.0
        max_width = max(1.0, float(self.config.gripper_max_width))
        self._gripper_percent = max_width if command > 0 else 0.0
        with self._sdk_lock:
            self.sdk.set_gripper_percent(
                self.config.gripper_width_addr,
                self.config.gripper_execute_addr,
                self._gripper_percent,
                force_percent=self.config.gripper_force_percent,
                execute_after_width=self.config.gripper_execute_after_width,
                max_width=self.config.gripper_max_width,
            )
        return command
