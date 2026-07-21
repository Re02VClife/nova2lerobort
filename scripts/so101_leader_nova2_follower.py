#!/usr/bin/env python3
"""
SO-101 → Nova2  关节映射遥操作 (v6 无阻力版)
============================================
纯关节空间相对映射, 无速度限制, 无MXD, 极轻平滑
夹爪: SO-101 ID=6 → Nova2 Modbus
用法: conda activate nova2_lerobot && python so101_leader_nova2_follower.py
"""

import sys, re, time, select, termios, tty
import serial

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

# Feetech
def _ck(d): return (~sum(d)&0xFF)
def _rd(ser,mid,addr,n):
    body=[mid,4,2,addr,n]
    ser.write(bytes([0xFF,0xFF]+body+[_ck(body)])); ser.flush(); time.sleep(0.001)
    r=ser.read(7+n)
    if len(r)>=6+n and r[0]==0xFF and r[1]==0xFF and r[2]==mid and r[4]==0:
        return list(r[5:5+n])
    return None
def _wr(ser,mid,addr,data):
    params=[addr]+data; body=[mid,2+len(params),3]+params
    ser.write(bytes([0xFF,0xFF]+body+[_ck(body)])); ser.flush(); time.sleep(0.001)
def pos(ser,mid):
    d=_rd(ser,mid,0x38,2); return d[0]+(d[1]<<8) if d else None
def tqoff(ser,mid): _wr(ser,mid,0x28,[0])
def gk():
    if select.select([sys.stdin],[],[],0)[0]: return sys.stdin.read(1)
    return ''

# ============================================================
PORT='/dev/ttyACM0'; N2='192.168.5.1'; HZ=50       # ServoJ流式控制

# 关节映射: (SO-101名, 舵机ID, Nova2索引, 缩放, 方向)
# scale=0.04394 = (360/4096)*0.5 → 2:1
JM=[
    ("shoulder_pan",  1,0,0.04394,-1),  # J1 (方向反)
    ("shoulder_lift", 2,1,0.04394,+1),  # J2
    ("elbow_flex",    3,2,0.04394,+1),  # J3
    ("wrist_flex",    4,3,0.04394,+1),  # J4
    ("wrist_roll",    5,4,0.04394,+1),  # J5
]
GID=6

# 安全: 无关节限位, 无MXD, 无速度限制. 仅笛卡尔+Y保底
YL=-560
CART_LO=(-330, -540,  50); CART_HI=(300, -280, 210)
CART_NAME=["X","Y","Z"]

# 平滑: 目标值步长限速, 消除启停顿挫
SENSOR_EMA=0.5   # 轻量传感器滤波
MAX_STEP=2.5     # 每帧ServoJ目标最大变化量(°), 确保连续运动不追到
SOFT_WALL=0.35   # 边界回拉力 (越大越硬, 0=无阻力)

# 死区
DZ=1            # 极小死区, 消除过零跳变

# 夹爪: 1秒内角度变化>20°打开, <-20°关闭
GRIP_RATE=228       # 20° raw (20/360*4096≈228)
GRIP_W=40000; GRIP_E=40008; GRIP_F=40001

# ============================================================
def main():
    print("="*56)
    print("  SO-101 → Nova2  关节映射 (无阻力版)")
    print("="*56)
    print("  缩放2:1  无关节限位  无速度限制  极轻平滑")

    # SO-101
    print(f"\n[SO-101] {PORT}")
    ser=serial.Serial(PORT,1000000,timeout=0.003)
    for _,mid,_,_,_ in JM: tqoff(ser,mid)
    tqoff(ser,GID)
    print("[SO-101] ✓")

    # Nova2
    print(f"\n[Nova2] {N2}")
    d=DobotApiDashboard(N2,29999)
    mode=_m(d.RobotMode())
    if mode!=5: d.ClearError(); time.sleep(0.3); d.EnableRobot(); time.sleep(1.5); mode=_m(d.RobotMode())
    d.SpeedFactor(100)  # 全速, 由MAX_STEP控速
    print(f"[Nova2] Mode={mode} SpeedFactor=100%")

    gid=None
    try:
        d.sendRecvMsg('SetTool485(115200,"N",1)'); time.sleep(0.5)
        r=str(d.sendRecvMsg(f"ModbusCreate({N2},60000,1,1)"))
        mx=re.search(r'\{(\d+)\}',r)
        if mx:
            gid=int(mx.group(1))
            d.SetHoldRegs(gid,GRIP_F,1,"{50}","U16")
            d.SetHoldRegs(gid,GRIP_E,1,"{1}","U16"); time.sleep(2)
            print(f"[Gripper] ✓ idx={gid}")
    except Exception as e: print(f"[Gripper] ⚠ {e}")

    # Calibrate
    print("\n"+"="*56)
    print("  摆好位姿, 按 Enter")
    print("="*56)
    input()

    s0={}
    for name,mid,_,_,_ in JM:
        s0[name]=pos(ser,mid)
        print(f"  SO-101 {name:15s} = {s0[name]}")

    n0=_p(d.GetAngle())[:6]
    p0=_p(d.GetPose())
    print(f"  Nova2  J=[{', '.join(f'{j:.2f}' for j in n0)}]")
    print(f"  Nova2 EE=[{', '.join(f'{v:.1f}' for v in p0[:6])}]")

    # Nova2自动回中 (笛卡尔空间限位中点)
    cx=(CART_LO[0]+CART_HI[0])/2
    cy=(CART_LO[1]+CART_HI[1])/2
    cz=(CART_LO[2]+CART_HI[2])/2
    rx,ry,rz=p0[3],p0[4],p0[5]
    print(f"\n[Nova2] 自动回中 MovL({cx:.0f},{cy:.0f},{cz:.0f})...")
    d.MovL(cx,cy,cz,rx,ry,rz,0)
    for _ in range(80):
        time.sleep(0.1)
        if gk() in (' ','\x1b'): break
    nc=_p(d.GetAngle())[:6]
    n0=list(nc); ns=list(nc); nt=list(nc)
    print(f"  回中完成 J=[{', '.join(f'{j:.1f}' for j in nc)}]")

    fd=sys.stdin.fileno(); old_term=termios.tcgetattr(fd); tty.setraw(fd)
    dt=1.0/HZ; fr=0; er=0; sp={}; grp_last=0; gh=[]
    safe_ns=list(ns); xm=ym=zm=0
    last_sn=dict(s0)  # 防丢帧: 读取失败时用上次值

    print("\n"+"="*56)
    print("  空格=急停  r=重设  ESC=退出")
    print("="*56)

    try:
        while True:
            t0=time.perf_counter()

            k=gk()
            if k==' ': print("\n[急停]"); break
            if k in ('\x1b','\x03'): print("\n[退出]"); break
            if k=='r':
                for name,mid,_,_,_ in JM:
                    p=pos(ser,mid)
                    if p is not None: s0[name]=p; last_sn[name]=p
                try: nc=_p(d.GetAngle())[:6]
                except: pass
                n0=list(nc); ns=list(nc); nt=list(nc)
                safe_ns=list(nc)
                print("\n[重设] ✓")

            # ── sensor (读失败用上次值, 不丢帧) ──
            sn={}
            for name,mid,_,_,_ in JM:
                p=pos(ser,mid)
                if p is not None:
                    if name in sp: p=sp[name]*SENSOR_EMA+p*(1-SENSOR_EMA)
                    sp[name]=p; sn[name]=p; last_sn[name]=p
                else:
                    sn[name]=last_sn[name]  # 用上次已知值, 不丢帧

            # ── Nova2 位姿 ──
            try:
                nc=_p(d.GetAngle())[:6]
                pn=_p(d.GetPose()); xm=pn[0]; ym=pn[1]; zm=pn[2]
            except: pass

            # ── 关节映射 ──
            for name,mid,idx,sc,dr in JM:
                raw=sn[name]-s0[name]
                if raw>2048: raw-=4096
                elif raw<-2048: raw+=4096
                if abs(raw)<DZ: nt[idx]=n0[idx]; continue
                nt[idx]=n0[idx]+raw*sc*dr

            # 目标步长限速: ns追赶nt但每帧最多变MAX_STEP°, 消除启停顿挫
            for i in range(6):
                diff=nt[i]-ns[i]
                if   diff> MAX_STEP: diff= MAX_STEP
                elif diff<-MAX_STEP: diff=-MAX_STEP
                ns[i]+=diff

            # ── 笛卡尔边界 (软墙壁, 可向内拉回) ──
            oob=(xm<CART_LO[0] or xm>CART_HI[0] or
                 ym<CART_LO[1] or ym>CART_HI[1] or
                 zm<CART_LO[2] or zm>CART_HI[2])
            if oob:
                for i in range(6):
                    ns[i]=ns[i]*(1-SOFT_WALL)+safe_ns[i]*SOFT_WALL
                if fr%HZ==0:
                    print(f"\n  [边界] X={xm:.0f} Y={ym:.0f} Z={zm:.0f} ←拉回")
            else:
                safe_ns=list(ns)

            # ── 模式检查 ──
            if fr%120==0:
                try:
                    mode=_m(d.RobotMode())
                    if mode==9: d.ClearError(); time.sleep(0.2); d.EnableRobot(); time.sleep(0.8)
                    elif mode==7: d.EnableRobot(); time.sleep(0.3)
                    elif mode not in (5,7): print(f"\n[异常] Mode={mode}"); break
                except: pass

            # ── ServoJ (每帧发送, 不冻结) ──
            try: d.ServoJ(*ns); er=0
            except Exception as e:
                er+=1
                if er<=1 or er%200==0: print(f"\n[ServoJ] {e}")

            # ── 夹爪 (每5帧读一次, 1秒内变化率) ──
            if gid is not None and fr%5==0:
                gn=pos(ser,GID)
                if gn is not None:
                    now=time.perf_counter()
                    gh.append((now,gn))
                    while gh and now-gh[0][0]>1.0: gh.pop(0)
                    if len(gh)>=2:
                        dg=gn-gh[0][1]
                        if dg>GRIP_RATE and grp_last!=1:
                            d.SetHoldRegs(gid,GRIP_W,1,"{90}","U16"); grp_last=1
                        elif dg<-GRIP_RATE and grp_last!=-1:
                            d.SetHoldRegs(gid,GRIP_W,1,"{10}","U16"); grp_last=-1

            el=time.perf_counter()-t0
            if el<dt: time.sleep(dt-el)
            fr+=1

            # ── 显示 (1Hz) ──
            if fr%HZ==0:
                ds=[f"{int(sn.get(n,0)-s0.get(n,0)):+4d}" for n,_,_,_,_ in JM]
                js=" ".join(f"{j:6.1f}" for j in ns)
                gr=""
                if gid is not None and len(gh)>=2:
                    dg=gh[-1][1]-gh[0][1]; gr=f" GΔ:{dg:+4d}"
                cw=" ⚡边界" if oob else ""
                print(f"\r  Y={ym:7.0f} Z={zm:6.0f} | J:[{js}] | Δ:[{' '.join(ds)}]{gr}{cw}    ",end='')

    except KeyboardInterrupt: pass
    finally:
        termios.tcsetattr(fd,termios.TCSADRAIN,old_term)
        if gid is not None:
            try: d.sendRecvMsg(f"ModbusClose({gid})")
            except: pass
        for _,mid,_,_,_ in JM: tqoff(ser,mid)
        tqoff(ser,GID); ser.close()
        print("\n[结束] ✓")

if __name__=='__main__': main()
