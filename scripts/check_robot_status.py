#!/usr/bin/env python3
"""
Nova2 机器人状态检查脚本
功能：只读查询 RobotMode、报警、关节角、TCP 位姿
适配服务器地址：192.168.0.130

用法：
    conda run --no-capture-output -n nova2_lerobot python check_robot_status.py
"""

import sys
import os
import traceback

# ============================================================
# ★ 配置区：修改此处适配你的环境
# ============================================================
ROBOT_IP = "192.168.0.130"      # 机器人控制器 IP
DASHBOARD_PORT = 29999
# Dobot SDK 路径（Linux 目标机上的绝对路径）
SDK_PATH = "/home/jjk/桌面/Project/dwlerobot-nova2xbox/TCP-IP-Python-V4-main/TCP-IP-Python-V4-main"

# ============================================================
# 安全检查
# ============================================================
print("=" * 60)
print("  Nova2 机器人状态检查")
print(f"  目标控制器: {ROBOT_IP}:{DASHBOARD_PORT}")
print("=" * 60)

# 添加 SDK 路径
if SDK_PATH not in sys.path:
    sys.path.insert(0, SDK_PATH)

try:
    from dobot_api import DobotApiDashboard
except ImportError:
    print("[错误] 无法导入 dobot_api，请确认 SDK 路径正确:")
    print(f"       {SDK_PATH}")
    sys.exit(1)

# ============================================================
# 连接并查询
# ============================================================
try:
    print(f"\n[1/5] 连接 Dashboard ({ROBOT_IP}:{DASHBOARD_PORT})...")
    d = DobotApiDashboard(ROBOT_IP, DASHBOARD_PORT)
    print("      ✓ 连接成功")

    # --- RobotMode ---
    print("\n[2/5] 查询 RobotMode...")
    mode = d.RobotMode()
    mode_map = {
        1: "INIT",
        2: "BRAKE_OPEN",
        3: "DISABLED",
        4: "ENABLE",
        5: "RUNNING (就绪, 可操作)",
        6: "BACKDRIVE",
        7: "RESET",
        8: "SENSORLESS_DRIVE",
        9: "ALARM (报警!)",
        10: "PAUSED (暂停!)",
    }
    mode_str = mode_map.get(mode, f"未知模式({mode})")
    if mode == 5:
        print(f"      ✓ Mode = {mode} ({mode_str})")
    else:
        print(f"      ✗ Mode = {mode} ({mode_str})  <-- 需要处理!")

    # --- ErrorID ---
    print("\n[3/5] 查询报警...")
    errors = d.GetErrorID()
    if not errors or errors == []:
        print("      ✓ 无报警")
    else:
        error_map = {
            17: "逆解无解 — 停止Y负方向，恢复安全位",
            27: "接近肘奇异点 — 停止、清错、关节插补回SAFE_A",
        }
        print(f"      ✗ 报警: {errors}")
        for e in errors:
            desc = error_map.get(e, "未知错误")
            print(f"         Error {e}: {desc}")

    # --- 关节角 ---
    print("\n[4/5] 查询关节角...")
    angles = d.GetAngle()
    print(f"      Joints = {angles}")
    # 与 SAFE_A 对比
    SAFE_A_JOINTS = [77.9857, 51.3401, 31.1298, 16.1389, -90.4004, 3.8710]
    print(f"      SAFE_A = {SAFE_A_JOINTS}")

    # --- TCP 位姿 ---
    print("\n[5/5] 查询 TCP 位姿...")
    pose = d.GetPose()
    print(f"      TCP = {pose}")
    SAFE_A_TCP = [0.4264, -559.5232, 322.8317, 171.4368, -0.9740, 164.2178]
    print(f"      SAFE_A_TCP = {SAFE_A_TCP}")

    # --- Y坐标安全检查 ---
    if pose and len(pose) >= 2:
        y = pose[1]
        if y < -560:
            print(f"\n      ⚠ 警告: Y={y:.2f} 已超出安全下限 -560mm!")
        else:
            print(f"\n      ✓ Y={y:.2f} 在安全范围内 (≥ -560mm)")

    # ============================================================
    # 综合判断
    # ============================================================
    print("\n" + "=" * 60)
    print("  综合判断:")
    all_ok = True

    if mode != 5:
        print("  ✗ RobotMode ≠ 5，需要恢复操作")
        all_ok = False
    else:
        print("  ✓ RobotMode = 5 (就绪)")

    if errors and errors != []:
        print(f"  ✗ 存在报警: {errors}")
        all_ok = False
    else:
        print("  ✓ 无报警")

    if mode == 5 and not (errors and errors != []):
        print("\n  ✅ 机器人状态正常，可以开始数据采集！")
    else:
        print("\n  ❌ 请先按第10章恢复流程处理后再采集")

except Exception as e:
    print(f"\n[致命错误] 连接或查询失败: {e}")
    print(f"请检查:")
    print(f"  1. 机器人控制器 IP 是否为 {ROBOT_IP}")
    print(f"  2. 本机网络: ping {ROBOT_IP}")
    print(f"  3. Dashboard 端口 {DASHBOARD_PORT} 是否被占用")
    traceback.print_exc()
    sys.exit(1)
