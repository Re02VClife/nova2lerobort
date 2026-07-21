#!/usr/bin/env python3
"""
Nova2 慢速绕工作空间限位一周
============================
沿笛卡尔边界矩形走一圈, 空格/ESC随时打断
MovL阻塞执行, 到位后才走下一段

用法: conda activate nova2_lerobot && python nova2_boundary_tour.py
"""

import sys, re, time, select, termios, tty, math

sys.path.insert(0,'/home/jjk/桌面/Project/dwlerobot-nova2xbox/TCP-IP-Python-V4-main/TCP-IP-Python-V4-main')
from dobot_api import DobotApiDashboard

def _p(raw):
    m=re.search(r'\{([^}]*)\}',str(raw))
    if not m: return []
    o=[]
    for v in m.group(1).split(','):
        try: o.append(float(v.strip()))
        except ValueError: pass
    return o

def gk():
    if select.select([sys.stdin],[],[],0)[0]: return sys.stdin.read(1)
    return ''

# ============================================================
N2='192.168.5.1'
SPEED=5   # %
Z=128     # 中间高度

# 限位矩形四角 + 回起点
P=[
    (-242,-375), (398,-375), (398,-590), (-242,-590), (-242,-375)
]

def dist(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

# ============================================================
def main():
    print("="*56)
    print("  Nova2 绕限位一周  Speed=10%  Z=128")
    print("  空格打断, MovL到位后自动下一段")
    print("="*56)

    d=DobotApiDashboard(N2,29999)
    # ensure Mode=5
    raw=str(d.RobotMode())
    m=re.search(r'\{(\d+)\}',raw); mode=int(m.group(1)) if m else 0
    if mode!=5:
        d.ClearError(); time.sleep(0.5); d.EnableRobot(); time.sleep(2)
        raw=str(d.RobotMode())
        m=re.search(r'\{(\d+)\}',raw); mode=int(m.group(1)) if m else 0
    d.SpeedFactor(SPEED)
    print(f"Mode={mode} ✓")

    pose=_p(d.GetPose())
    if len(pose)<6: print("读取位姿失败"); return
    rx,ry,rz=pose[3],pose[4],pose[5]
    print(f"起点 X={pose[0]:.0f} Y={pose[1]:.0f} Z={pose[2]:.0f}")
    print(f"保持姿态 rx={rx:.1f} ry={ry:.1f} rz={rz:.1f}")
    total=sum(dist(P[i],P[i+1]) for i in range(len(P)-1))
    print(f"总行程 ≈ {total:.0f}mm")

    fd=sys.stdin.fileno(); old=termios.tcgetattr(fd); tty.setraw(fd)

    print("\n按 Enter 开始, 空格打断")
    try:
        while True:
            k=gk()
            if k in ('\n','\r'): break
            if k in (' ','\x1b','\x03'): print("\n取消"); return
            time.sleep(0.05)

        stopped=False
        for i in range(len(P)):
            tx,ty=P[i]

            # 检查空格
            k=gk()
            if k in (' ','\x1b','\x03'):
                print(f"\n打断 ({i}/{len(P)-1})"); stopped=True; break

            # 刷新当前位姿(取最新rx/ry/rz)
            cur=_p(d.GetPose())
            if len(cur)>=6: rx,ry,rz=cur[3],cur[4],cur[5]

            segment=f"[{i+1}/{len(P)-1}]" if i<len(P)-1 else "[回起点]"
            print(f"\n{segment} MovL → X={tx:.0f} Y={ty:.0f}")

            try:
                # MovL(x,y,z,rx,ry,rz, coordMode=0)
                d.MovL(float(tx),float(ty),float(Z),rx,ry,rz,0)
            except Exception as e:
                print(f"  错误: {e}"); break

            # 等MovL完成: 检查是否到达目标
            timeout=time.perf_counter()+60
            arrived=False
            while time.perf_counter()<timeout:
                time.sleep(0.3)
                k=gk()
                if k in (' ','\x1b','\x03'):
                    print("  打断!"); stopped=True; break
                c=_p(d.GetPose())
                if len(c)>=6:
                    dxy=math.sqrt((c[0]-tx)**2+(c[1]-ty)**2)
                    dz=abs(c[2]-Z)
                    arrived=dxy<10 and dz<5
                    if arrived: break
            if stopped: break

            if arrived:
                print(f"  ✓ 到位, 停5秒...")
                for _ in range(50):  # 5s, 每0.1s检查打断
                    time.sleep(0.1)
                    if gk() in (' ','\x1b','\x03'):
                        print(" 打断!"); stopped=True; break
                if stopped: break
            else:
                print(f"  ⚠ 可能未到位, 继续下一段")

        if not stopped: print("\n✓ 完成一圈")

    finally:
        termios.tcsetattr(fd,termios.TCSADRAIN,old)
        print("[结束]")

if __name__=='__main__': main()
