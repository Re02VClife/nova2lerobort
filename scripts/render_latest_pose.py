#!/usr/bin/env python3
"""用最新数据集抓取姿态渲染仿真 front 相机,与真实顶视图对比。"""
import os, re
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from PIL import Image
from sim_smolvla_rollout import (SimNova2, R_BASE, T_BASE, GRIPPER_OPEN_M,
                                 zyx_euler_to_rot)

XML = "/home/jjk/dobot_nova2_project/sim/nova2_scene_corrected.xml"

# ep0 抓取: tcp(-138.1,-469.7,31.4) rpy(-175.37,-2.5,171.11) 夹爪开
tcp_real = np.array([-138.1, -469.7, 31.4])
rpy = np.array([-175.37, -2.5, 171.11])
# 方块: tcp xy, 30mm高方块放在桌面上(CAD z中心 -0.079)
block_real = np.array([-138.1, -469.7, 79.0])  # 79mm real -> CAD ≈ -0.078
block_cad = R_BASE @ (block_real * 1e-3 - T_BASE * 1e-3)
block_cad[1] += 0.10  # 工作区前移10cm(机械臂远离工作台)
block_cad[2] = -0.079
tcp_cad = R_BASE @ (tcp_real * 1e-3 - T_BASE * 1e-3)
tcp_cad[1] += 0.10  # 工作区前移10cm
tcp_mat = R_BASE @ zyx_euler_to_rot(rpy)

s = SimNova2(XML)
s.reset()
mujoco.mj_forward(s.model, s.data)
s.set_block_pos_cad(block_cad)
q = s.ik_to_q(tcp_cad, tcp_mat)
s.data.qpos[:6] = q
fq = s.model.joint("finger_left_joint").qposadr[0]
s.data.qpos[fq] = GRIPPER_OPEN_M
s.data.qpos[fq + 1] = GRIPPER_OPEN_M
mujoco.mj_forward(s.model, s.data)
img = s.render("front")
Image.fromarray(img).save("/tmp/latest_sim_front.png")
R, G, B = img[..., 0].astype(float), img[..., 1].astype(float), img[..., 2].astype(float)
blk = (R > 150) & (G > 100) & (G < 200) & (B < 120) & (R - G > 30)
if blk.sum() > 30:
    yy, xx = np.where(blk)
    print(f"sim 方块+垫: cx={xx.mean():.0f} cy={yy.mean():.0f} w={xx.max()-xx.min()} h={yy.max()-yy.min()}")
dark = (R < 80) & (G < 80) & (B < 80)
if dark.sum() > 1000:
    yy, xx = np.where(dark)
    print(f"sim 深色(arm): cx={xx.mean():.0f} cy={yy.mean():.0f} w={xx.max()-xx.min()} h={yy.max()-yy.min()}")
