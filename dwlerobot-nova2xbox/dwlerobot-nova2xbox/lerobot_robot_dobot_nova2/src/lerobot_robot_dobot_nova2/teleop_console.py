from __future__ import annotations

import argparse
import time

from .config_dobot_nova2 import DobotNova2Config, DobotNova2KeyboardConfig
from .dobot_nova2 import DobotNova2
from .keyboard_teleop import DobotNova2Keyboard


def main() -> None:
    parser = argparse.ArgumentParser(description="DOBOT Nova2 控制台键盘遥操")
    parser.add_argument("--host", default="192.168.5.1")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--startup", action="store_true")
    parser.add_argument("--power-on-wait", type=float, default=10.0)
    parser.add_argument("--enable-gripper", action="store_true")
    parser.add_argument("--gripper-modbus-ip", default="127.0.0.1")
    parser.add_argument("--gripper-modbus-port", type=int, default=60000)
    parser.add_argument("--gripper-slave-id", type=int, default=1)
    parser.add_argument("--gripper-force-percent", type=int, default=50)
    parser.add_argument("--gripper-execute-after-width", action="store_true")
    parser.add_argument("--control-mode", choices=["movl", "movj", "servop"], default="movl")
    parser.add_argument("--speed-factor", type=int, default=5)
    parser.add_argument("--xyz-step", type=float, default=1.0)
    parser.add_argument("--rot-step", type=float, default=0.5)
    args = parser.parse_args()

    robot = DobotNova2(
        DobotNova2Config(
            host=args.host,
            startup_on_connect=args.startup,
            startup_power_on_wait_s=args.power_on_wait,
            enable_gripper=args.enable_gripper,
            gripper_modbus_ip=args.gripper_modbus_ip,
            gripper_modbus_port=args.gripper_modbus_port,
            gripper_slave_id=args.gripper_slave_id,
            gripper_force_percent=args.gripper_force_percent,
            gripper_execute_after_width=args.gripper_execute_after_width,
            control_mode=args.control_mode,
            speed_factor=args.speed_factor,
            max_xyz_step_mm=args.xyz_step,
            max_rot_step_deg=args.rot_step,
        )
    )
    teleop = DobotNova2Keyboard(DobotNova2KeyboardConfig(xyz_step=args.xyz_step, rot_step=args.rot_step))
    robot.connect()
    teleop.connect()
    try:
        period = 1.0 / max(args.fps, 1.0)
        while True:
            action = teleop.get_action()
            if any(abs(float(v)) > 1e-9 for v in action.values()):
                sent = robot.send_action(action)
                print("sent", sent)
            time.sleep(period)
    except KeyboardInterrupt:
        print("退出 Nova2 遥操")
    finally:
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
