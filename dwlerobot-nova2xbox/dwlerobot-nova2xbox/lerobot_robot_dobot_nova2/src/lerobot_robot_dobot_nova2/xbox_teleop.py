from __future__ import annotations

import time
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import require_package

from .config_dobot_nova2 import DobotNova2XboxConfig


class DobotNova2Xbox(Teleoperator):
    """Xbox 手柄遥操器，输出 Nova2 的 7 维增量动作。"""

    config_class = DobotNova2XboxConfig
    name = "dobot_nova2_xbox"

    def __init__(self, config: DobotNova2XboxConfig):
        super().__init__(config)
        self.config = config
        self._pygame: Any | None = None
        self._joystick: Any | None = None
        self._connected = False
        self._success_requested = False
        self._rerecord_requested = False
        self._stop_requested = False
        self._home_requested = False
        self._axis_centered_until: dict[int, float] = {}
        self._axis_last_emit: dict[int, float] = {}
        self._axis_last_sign: dict[int, float] = {}
        self._axis_last_magnitude: dict[int, float] = {}
        self._axis_releasing: dict[int, bool] = {}
        self._axis_filtered: dict[int, float] = {}

    @property
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
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        require_package("pygame", extra="gamepad")
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            pygame.quit()
            raise RuntimeError("没有检测到 Xbox/pygame 手柄，请先连接手柄。")

        self._pygame = pygame
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        self._connected = True
        print(f"Nova2 Xbox 遥操已连接：{self._joystick.get_name()}")
        print(
            "左摇杆 X/Y 平移，右摇杆上下 Z，方向键左右 Rz，LB/RB Ry，LT/RT 夹爪关/开，"
            "Y 回初始位置，A 成功结束，X 重录，B 停止。"
        )

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        pygame = self._require_pygame()
        joystick = self._require_joystick()
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == self.config.a_button:
                    self._success_requested = True
                elif event.button == self.config.x_button:
                    self._rerecord_requested = True
                elif event.button == self.config.b_button:
                    self._stop_requested = True
                elif event.button == self.config.y_button:
                    self._home_requested = True

        if self.config.control_style.strip().lower() == "step":
            left_x = self._step_axis(joystick, self.config.left_x_axis, self.config.invert_left_x)
            left_y = self._step_axis(joystick, self.config.left_y_axis, self.config.invert_left_y)
            right_y = self._step_axis(joystick, self.config.right_y_axis, self.config.invert_right_y)
        else:
            left_x = self._smooth_axis(joystick, self.config.left_x_axis, self.config.invert_left_x)
            left_y = self._smooth_axis(joystick, self.config.left_y_axis, self.config.invert_left_y)
            right_y = self._smooth_axis(joystick, self.config.right_y_axis, self.config.invert_right_y)

        dpad_x = self._hat_x(joystick)
        gripper = 0.0
        if self._button(joystick, self.config.lb_button):
            delta_ry = self.config.rot_step
        elif self._button(joystick, self.config.rb_button):
            delta_ry = -self.config.rot_step
        else:
            delta_ry = 0.0

        lt = self._trigger(joystick, self.config.lt_axis)
        rt = self._trigger(joystick, self.config.rt_axis)
        if lt > self.config.trigger_threshold and rt <= self.config.trigger_threshold:
            gripper = -1.0
        elif rt > self.config.trigger_threshold and lt <= self.config.trigger_threshold:
            gripper = 1.0

        return {
            "delta_x": left_x * self.config.xyz_step,
            "delta_y": -left_y * self.config.xyz_step,
            "delta_z": -right_y * self.config.xyz_step,
            "delta_rx": 0.0,
            "delta_ry": delta_ry,
            "delta_rz": -dpad_x * self.config.rot_step,
            "gripper": gripper,
        }

    def get_teleop_events(self) -> dict[Any, bool]:
        success = self._success_requested
        rerecord = self._rerecord_requested
        stop = self._stop_requested
        home = self._home_requested
        self._success_requested = False
        self._rerecord_requested = False
        self._stop_requested = False
        self._home_requested = False
        return {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.TERMINATE_EPISODE: success or rerecord or stop,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord,
            "stop_recording": stop,
            "home_position": home,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._joystick is not None:
            self._joystick.quit()
            self._joystick = None
        if self._pygame is not None:
            self._pygame.joystick.quit()
            self._pygame.quit()
            self._pygame = None
        self._connected = False

    def _require_pygame(self) -> Any:
        if self._pygame is None:
            raise RuntimeError("pygame 尚未连接")
        return self._pygame

    def _require_joystick(self) -> Any:
        if self._joystick is None:
            raise RuntimeError("Xbox 手柄尚未连接")
        return self._joystick

    def _axis_value(self, joystick: Any, axis: int, invert: bool) -> float:
        value = float(joystick.get_axis(axis)) if joystick.get_numaxes() > axis else 0.0
        if abs(value) < self.config.deadzone:
            return 0.0
        sign = -1.0 if value < 0 else 1.0
        magnitude = (abs(value) - self.config.deadzone) / (1.0 - self.config.deadzone)
        magnitude = max(0.0, min(1.0, magnitude)) ** max(1.0, float(self.config.axis_curve))
        magnitude = max(min(1.0, float(self.config.min_output)), magnitude)
        curved = sign * magnitude
        return -curved if invert else curved

    def _step_axis(self, joystick: Any, axis: int, invert: bool) -> float:
        raw = self._axis_value(joystick, axis, invert)
        now = time.perf_counter()
        if raw == 0.0:
            self._axis_centered_until[axis] = now + max(0.0, float(self.config.center_release_s))
            self._axis_last_sign[axis] = 0.0
            self._axis_last_magnitude[axis] = 0.0
            self._axis_releasing[axis] = False
            return 0.0

        sign = -1.0 if raw < 0 else 1.0
        if now < self._axis_centered_until.get(axis, 0.0) and abs(raw) < self.config.release_deadzone:
            return 0.0

        last_sign = self._axis_last_sign.get(axis, 0.0)
        if last_sign not in (0.0, sign):
            return 0.0
        magnitude = abs(raw)
        last_magnitude = self._axis_last_magnitude.get(axis, 0.0)
        if self._axis_releasing.get(axis, False):
            self._axis_last_magnitude[axis] = magnitude
            return 0.0
        if last_sign == sign and last_magnitude > 0.65 and magnitude < last_magnitude - max(0.0, float(self.config.release_drop)):
            self._axis_releasing[axis] = True
            self._axis_last_magnitude[axis] = magnitude
            return 0.0
        last_emit = self._axis_last_emit.get(axis, -1e9)
        if last_sign != sign or now - last_emit >= max(0.01, float(self.config.repeat_s)):
            self._axis_last_sign[axis] = sign
            self._axis_last_emit[axis] = now
            self._axis_last_magnitude[axis] = magnitude
            if magnitude >= self.config.turbo_threshold:
                magnitude *= max(1.0, float(self.config.turbo_multiplier))
            return sign * magnitude
        self._axis_last_magnitude[axis] = magnitude
        return 0.0

    def _smooth_axis(self, joystick: Any, axis: int, invert: bool) -> float:
        target = self._axis_value(joystick, axis, invert)
        previous = self._axis_filtered.get(axis, 0.0)
        alpha = self.config.stop_smoothing if target == 0.0 else self.config.smoothing
        alpha = max(0.0, min(1.0, float(alpha)))
        filtered = previous + alpha * (target - previous)
        if abs(filtered) < 1e-3:
            filtered = 0.0
        self._axis_filtered[axis] = filtered
        return filtered

    @staticmethod
    def _hat_x(joystick: Any, hat: int = 0) -> float:
        if joystick.get_numhats() <= hat:
            return 0.0
        x, _y = joystick.get_hat(hat)
        return float(x)

    @staticmethod
    def _button(joystick: Any, button: int) -> bool:
        return bool(joystick.get_button(button)) if joystick.get_numbuttons() > button else False

    @staticmethod
    def _trigger(joystick: Any, axis: int) -> float:
        if joystick.get_numaxes() <= axis:
            return 0.0
        return (float(joystick.get_axis(axis)) + 1.0) / 2.0
