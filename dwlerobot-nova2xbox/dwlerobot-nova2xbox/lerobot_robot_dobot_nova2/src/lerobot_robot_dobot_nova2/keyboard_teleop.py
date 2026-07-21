from __future__ import annotations

import msvcrt
from threading import Lock
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_dobot_nova2 import DobotNova2KeyboardConfig


class DobotNova2Keyboard(Teleoperator):
    """Windows 控制台键盘遥操器，输出 Nova2 的 7 维增量动作。"""

    config_class = DobotNova2KeyboardConfig
    name = "dobot_nova2_keyboard"

    def __init__(self, config: DobotNova2KeyboardConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._listener: Any | None = None
        self._lock = Lock()
        self._pending_keys: list[str] = []
        self._exit_requested = False

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
        self._connected = True
        self._start_global_listener()
        print("Nova2 键盘遥操已连接：W/S Y, A/D X, R/F Z, I/K Rx, J/L Ry, Q/E Rz, [/ ] 夹爪, T 退出当前脚本。")

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action = dict.fromkeys(self.action_features, 0.0)
        for key in self._pop_keys():
            self._apply_key(action, key)
        if self._exit_requested:
            self._exit_requested = False
            raise KeyboardInterrupt("用户按 T 退出 Nova2 键盘遥操")
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._connected = False

    def _start_global_listener(self) -> None:
        try:
            from pynput import keyboard
        except Exception:
            return

        def on_press(key: Any) -> None:
            char = getattr(key, "char", None)
            if not char:
                return
            with self._lock:
                self._pending_keys.append(char.lower())

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

    def _pop_keys(self) -> list[str]:
        keys: list[str] = []
        if self._listener is None:
            while msvcrt.kbhit():
                raw = msvcrt.getch()
                if raw in (b"\x00", b"\xe0"):
                    msvcrt.getch()
                    continue
                keys.append(raw.decode("utf-8", errors="ignore").lower())
            return keys

        with self._lock:
            keys = self._pending_keys
            self._pending_keys = []
        return keys

    def _apply_key(self, action: RobotAction, key: str) -> None:
        if key == "w":
            action["delta_y"] -= self.config.xyz_step
        elif key == "s":
            action["delta_y"] += self.config.xyz_step
        elif key == "a":
            action["delta_x"] += self.config.xyz_step
        elif key == "d":
            action["delta_x"] -= self.config.xyz_step
        elif key == "r":
            action["delta_z"] += self.config.xyz_step
        elif key == "f":
            action["delta_z"] -= self.config.xyz_step
        elif key == "i":
            action["delta_rx"] += self.config.rot_step
        elif key == "k":
            action["delta_rx"] -= self.config.rot_step
        elif key == "j":
            action["delta_ry"] += self.config.rot_step
        elif key == "l":
            action["delta_ry"] -= self.config.rot_step
        elif key in ("u", "q"):
            action["delta_rz"] += self.config.rot_step
        elif key in ("o", "e"):
            action["delta_rz"] -= self.config.rot_step
        elif key == "]":
            action["gripper"] = 1.0
        elif key == "[":
            action["gripper"] = -1.0
        elif key == "t":
            self._exit_requested = True
