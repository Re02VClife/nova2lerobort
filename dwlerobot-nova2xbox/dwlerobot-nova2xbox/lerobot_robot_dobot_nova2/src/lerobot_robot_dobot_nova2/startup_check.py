from __future__ import annotations

import argparse
import socket
from contextlib import closing

from .config_dobot_nova2 import DEFAULT_DOBOT_SDK_PATH
from .dobot_sdk_adapter import DobotNova2OfficialSDK


def port_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with closing(socket.create_connection((host, port), timeout=timeout_s)):
            return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Nova2 上电使能和端口检测")
    parser.add_argument("--host", default="192.168.5.1")
    parser.add_argument("--dashboard-port", type=int, default=29999)
    parser.add_argument("--move-port", type=int, default=29999)
    parser.add_argument("--speed-factor", type=int, default=5)
    parser.add_argument("--power-on-wait", type=float, default=10.0)
    parser.add_argument("--no-startup", action="store_true", help="只读状态和探测端口，不执行清错/上电/使能")
    args = parser.parse_args()

    robot = DobotNova2OfficialSDK(
        args.host,
        args.dashboard_port,
        args.move_port,
        DEFAULT_DOBOT_SDK_PATH,
        speed_factor=args.speed_factor,
    )
    robot.connect()
    try:
        print(f"dashboard {args.dashboard_port}: connected")
        if not args.no_startup:
            print("执行使能流程：ClearError -> PowerOn -> EnableRobot -> SpeedFactor")
            robot.startup(power_on_wait_s=args.power_on_wait)
        try:
            print("robot_mode=", robot.robot_mode())
        except Exception as exc:
            print("robot_mode 读取失败:", exc)
        print("joints=", robot.get_joints())
        print("pose=", robot.get_pose())
    finally:
        robot.close()

    for port in dict.fromkeys((args.dashboard_port, args.move_port, 30004, 30005)):
        print(f"port {port} open=", port_open(args.host, port))


if __name__ == "__main__":
    main()