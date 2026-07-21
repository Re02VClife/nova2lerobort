#!/usr/bin/env python3
"""
SSH 键盘发送器 — 终端 B
通过 UDP 发送键盘控制指令到录制脚本（终端 A）

运行方式（在终端 B）：
    conda run --no-capture-output -n nova2_lerobot python ssh_key_sender.py

按键映射：
    运动控制:
        W/S      — Y- / Y+     (W = 远离底座，危险方向!)
        A/D      — X+ / X-
        R/F      — Z+ / Z-     (R = 抬高, F = 下降)
        I/K      — Rx+ / Rx-
        J/L      — Ry+ / Ry-
        Q/E      — Rz+ / Rz-
        空格      — 停止动作，立即归零
        1/2/3     — 精调 / 正常 / 粗调

    夹爪控制:
        ]        — 张开夹爪
        [        — 闭合夹爪

    录制控制:
        n        — 保存当前 episode（成功轨迹）
        m        — 丢弃当前 episode 并重录（失败轨迹）
        P        — 正常结束整批（仅在确需提前终止时使用）

    注意：
    - 本脚本只监听本终端标准输入，确保焦点在本终端
    - 连续长按会产生连续动作，接近边界时短按并频繁按空格归零
    - 不要在录制过程中按 Ctrl+C
"""

import socket
import sys
import termios
import tty
import select
import os

# ============================================================
# ★ 配置
# ============================================================
UDP_HOST = "127.0.0.1"
UDP_PORT = 18888

# 按键映射到控制字符（保持与 LeRobot 插件约定一致）
KEY_MAP = {
    # ---- 平移 ----
    'w': 'w',    # Y-
    's': 's',    # Y+
    'a': 'a',    # X+
    'd': 'd',    # X-
    'r': 'r',    # Z+
    'f': 'f',    # Z-

    # ---- 旋转 ----
    'i': 'i',    # Rx+
    'k': 'k',    # Rx-
    'j': 'j',    # Ry+
    'l': 'l',    # Ry-
    'q': 'q',    # Rz+
    'e': 'e',    # Rz-

    # ---- 归零 ----
    ' ': ' ',    # 停止动作

    # ---- 速度档位 ----
    '1': '1',    # 精调
    '2': '2',    # 正常
    '3': '3',    # 粗调

    # ---- 夹爪 ----
    ']': ']',    # 张开
    '[': '[',    # 闭合

    # ---- 录制控制 ----
    'n': 'n',    # 保存 episode
    'm': 'm',    # 丢弃并重录
    'P': 'P',    # 正常结束整批

    # ---- 退出本发送器 ----
    '\x1b': '\x1b',  # ESC 退出
}

# 按键说明
HELP_TEXT = """
╔══════════════════════════════════════════════════════════╗
║            Nova2 SSH 键盘发送器 (终端 B)                ║
║            服务器: 192.168.0.130                        ║
╠══════════════════════════════════════════════════════════╣
║  运动控制:                                              ║
║    W/S   — Y-/Y+  (W远离底座, 危险!)                    ║
║    A/D   — X+/X-                                        ║
║    R/F   — Z+/Z-  (R抬高 / F下降)                       ║
║    I/K   — Rx+/Rx-                                      ║
║    J/L   — Ry+/Ry-                                      ║
║    Q/E   — Rz+/Rz-                                      ║
║    空格   — 停止动作, 归零                              ║
║    1/2/3 — 精调 / 正常 / 粗调                           ║
╠══════════════════════════════════════════════════════════╣
║  夹爪:                                                  ║
║    ]     — 张开夹爪                                     ║
║    [     — 闭合夹爪                                     ║
╠══════════════════════════════════════════════════════════╣
║  录制控制:                                              ║
║    n     — ✓ 保存当前成功 episode                       ║
║    m     — ✗ 丢弃失败 episode 并重录                    ║
║    P     — ■ 正常结束整批                               ║
╠══════════════════════════════════════════════════════════╣
║  ESC    — 退出本发送器                                  ║
╚══════════════════════════════════════════════════════════╝
"""


def get_key() -> str:
    """非阻塞读取单个按键"""
    if select.select([sys.stdin], [], [], 0.05)[0]:
        return sys.stdin.read(1)
    return ''


def main():
    print("=" * 56)
    print("  Nova2 SSH 键盘发送器")
    print(f"  UDP 目标: {UDP_HOST}:{UDP_PORT}")
    print("  按 n=保存, m=丢弃重录, P=结束整批, ESC=退出")
    print("=" * 56)
    print(HELP_TEXT)

    # 保存终端原始设置
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # 创建 UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # 设置终端为 raw 模式（无缓冲、无回显）
        tty.setraw(fd)

        print(">>> 发送器已就绪，等待按键...\n")

        while True:
            key = get_key()
            if not key:
                continue

            # ESC 退出
            if key == '\x1b':
                print("\n>>> 发送器正常退出")
                break

            # Ctrl+C 退出
            if key == '\x03':
                print("\n>>> 发送器正常退出 (Ctrl+C)")
                break

            # 发送已知按键
            if key in KEY_MAP:
                sock.sendto(key.encode('utf-8'), (UDP_HOST, UDP_PORT))

                # 打印反馈
                labels = {
                    'n': '✓ 保存 episode',
                    'm': '✗ 丢弃重录',
                    'P': '■ 结束整批',
                    ' ': '■ 停止归零',
                    '1': '精调', '2': '正常', '3': '粗调',
                    '[': '闭合夹爪', ']': '张开夹爪',
                }
                label = labels.get(key, f"按键 '{key}'")
                print(f"\r>>> {label}                          ")
            # 忽略其他未知按键

    except KeyboardInterrupt:
        print("\n>>> 发送器正常退出 (Ctrl+C)")
    except Exception as e:
        print(f"\n[错误] {e}")
    finally:
        # 恢复终端设置
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sock.close()
        print(">>> UDP socket 已关闭")


if __name__ == "__main__":
    main()
