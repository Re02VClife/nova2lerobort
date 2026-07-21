from __future__ import annotations

import time


def main() -> None:
    import pygame

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        raise RuntimeError("没有检测到手柄，请先连接 Xbox 手柄。")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"手柄: {joystick.get_name()}")
    print(f"轴数量: {joystick.get_numaxes()}, 按钮数量: {joystick.get_numbuttons()}, Hat 数量: {joystick.get_numhats()}")
    print("移动摇杆、按 LT/RT/LB/RB/A/B/X，观察哪个 axis/button 在变化。Ctrl+C 退出。")
    try:
        while True:
            pygame.event.pump()
            axes = [round(float(joystick.get_axis(i)), 3) for i in range(joystick.get_numaxes())]
            buttons = [int(joystick.get_button(i)) for i in range(joystick.get_numbuttons())]
            hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
            print(f"\raxes={axes} buttons={buttons} hats={hats}", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n退出手柄诊断。")
    finally:
        joystick.quit()
        pygame.joystick.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
