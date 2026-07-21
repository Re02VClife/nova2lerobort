#!/usr/bin/env python3
"""
Nova2 键盘遥控标定工作空间边界
==============================
键盘移动机械臂到边界角点, 按 Enter 记录
按 P 结束并输出 CART_LO / CART_HI 配置

按键:
  W/S  Y-/Y+    A/D  X+/X-     R/F  Z+/Z-
  1/2  速度     空格=归零停止   Enter=记录角点   P=结束输出

用法: conda activate nova2_lerobot && python nova2_boundary_calibrate.py
"""

import sys, re, time, select, termios, tty

sys.path.insert(0,'/home/jjk/桌面/Project/dwlerobot-nova2xbox/TCP-IP-Python-V4-main/TCP-IP-Python-V4-main')
from dobot_api import DobotApiDashboard, DobotApi

def _p(raw):
    m=re.search(r'\{([^}]*)\}',str(raw))
    if not m: return []
    o=[]
    for v in m.group(1).split(','):
        try: o.append(float(v.strip()))
        except ValueError: pass
    return o
def _m(raw):
    v=_p(raw); return int(v[0]) if v else 0

def gk():
    if select.select([sys.stdin],[],[],0)[0]: return sys.stdin.read(1)
    return ''

# ============================================================
N2='192.168.5.1'; HZ=20
STEP_XY=5.0; STEP_Z=3.0  # mm per keypress

def main():
    print("="*56)
    print("  Nova2 边界标定 — 键盘遥控")
    print("="*56)
    print("  W/S Y轴  A/D X轴  R/F Z轴")
    print("  1=慢(3mm)  2=快(10mm)")
    print("  Enter=记录角点  P=结束输出")
    print("="*56)

    d=DobotApiDashboard(N2,29999)
    m=DobotApi(N2,30004)
    mode=_m(d.RobotMode())
    if mode!=5: d.ClearError(); time.sleep(0.3); d.EnableRobot(); time.sleep(1.5); mode=_m(d.RobotMode())
    d.SpeedFactor(20)
    print(f"Mode={mode} ✓")

    fd=sys.stdin.fileno(); old=termios.tcgetattr(fd); tty.setraw(fd)
    corners=[]
    speed_idx=0  # 0=slow, 1=fast
    step_xy=[3.0,10.0]; step_z=[2.0,6.0]

    print("\n  移动到位后按 Enter 记录角点")
    print("  至少记录2个对角点, 按 P 结束\n")

    try:
        while True:
            k=gk()
            if not k:
                time.sleep(0.05)
                continue

            if k in ('\x1b','\x03'): print("\n退出\n"); break
            if k=='P' or k=='p':
                if len(corners)>=2:
                    xs=[c[0] for c in corners]; ys=[c[1] for c in corners]; zs=[c[2] for c in corners]
                    lo_x,hi_x=min(xs),max(xs); lo_y,hi_y=min(ys),max(ys); lo_z,hi_z=min(zs),max(zs)
                    # +5mm margin
                    lo_x-=5; lo_y-=5; lo_z-=5; hi_x+=5; hi_y+=5; hi_z+=5
                    print(f"\n{'='*56}")
                    print(f"  角点: {len(corners)}个")
                    for i,c in enumerate(corners):
                        print(f"    [{i}] X={c[0]:.0f} Y={c[1]:.0f} Z={c[2]:.0f}")
                    print(f"\n  CART_LO=({lo_x:.0f}, {lo_y:.0f}, {lo_z:.0f})")
                    print(f"  CART_HI=({hi_x:.0f}, {hi_y:.0f}, {hi_z:.0f})")
                    print(f"{'='*56}\n")
                else:
                    print("\n  ⚠ 至少2个角点!\n")
                break

            if k=='\n' or k=='\r':
                pose=_p(d.GetPose())
                if len(pose)>=6:
                    corners.append((pose[0],pose[1],pose[2]))
                    print(f"\r  ✓ 记录 #{len(corners)}: X={pose[0]:.0f} Y={pose[1]:.0f} Z={pose[2]:.0f}          \n")
                continue

            if k=='1': speed_idx=0; print(f"\r  慢速: XY={step_xy[0]}mm Z={step_z[0]}mm          "); continue
            if k=='2': speed_idx=1; print(f"\r  快速: XY={step_xy[1]}mm Z={step_z[1]}mm          "); continue
            if k==' ':
                # stop (归零动作)
                continue

            pose=_p(d.GetPose())
            if len(pose)<6: continue
            x,y,z,rx,ry,rz=pose[:6]
            dxy=step_xy[speed_idx]; dz=step_z[speed_idx]
            moved=False

            if k=='w': y-=dxy; moved=True
            if k=='s': y+=dxy; moved=True
            if k=='a': x+=dxy; moved=True
            if k=='d': x-=dxy; moved=True
            if k=='r': z+=dz;  moved=True
            if k=='f': z-=dz;  moved=True

            if moved:
                try:
                    d.MovL(x,y,z,rx,ry,rz,0)
                    cur=_p(d.GetPose())
                    if len(cur)>=6:
                        print(f"\r  X={cur[0]:6.0f} Y={cur[1]:6.0f} Z={cur[2]:6.0f}  [{len(corners)}个角点]    ",end='')
                except Exception as e:
                    print(f"\r  [错误] {e}")

    finally:
        termios.tcsetattr(fd,termios.TCSADRAIN,old)
        print("[结束]")

if __name__=='__main__': main()
