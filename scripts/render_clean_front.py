#!/usr/bin/env python3
"""干净渲染当前 front 相机(无染色)在真实 ep0 hover 姿态下的画面,保存 PNG。"""
import os, re
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
from PIL import Image
from sim_smolvla_rollout import (SimNova2, R_BASE, T_BASE, GRIPPER_OPEN_M,
                                 zyx_euler_to_rot)

XML = "/home/jjk/dobot_nova2_project/sim/nova2_scene_corrected.xml"

block_real = np.array([-112.0, -379.0, 70.0])
block_cad = R_BASE @ (block_real * 1e-3 - T_BASE * 1e-3)
tcp_real = np.array([-112.0, -379.0, 64.2])
rpy = np.array([-179.7, 2.6, 165.8])
tcp_cad = R_BASE @ (tcp_real * 1e-3 - T_BASE * 1e-3)
tcp_mat = R_BASE @ zyx_euler_to_rot(rpy)

s = SimNova2(XML)
s.reset()
import mujoco
mujoco.mj_forward(s.model, s.data)
s.set_block_pos_cad(block_cad)
q = s.ik_to_q(tcp_cad, tcp_mat)
s.data.qpos[:6] = q
fq = s.model.joint("finger_left_joint").qposadr[0]
s.data.qpos[fq] = GRIPPER_OPEN_M
s.data.qpos[fq + 1] = GRIPPER_OPEN_M
mujoco.mj_forward(s.model, s.data)
img = s.render("front")
Image.fromarray(img).save("/tmp/front_clean.png")
R, G, B = img[..., 0].astype(float), img[..., 1].astype(float), img[..., 2].astype(float)
blk = (R > 150) & (G > 90) & (G < 200) & (B < 120) & (R - G > 30)
if blk.sum() > 30:
    yy, xx = np.where(blk)
    print(f"block(pad+方块黄): cx={xx.mean():.0f} cy={yy.mean():.0f} w={xx.max()-xx.min()} h={yy.max()-yy.min()}")
red = (R > 90) & (G < 70) & (B < 70)
if red.sum() > 100:
    yy, xx = np.where(red)
    print(f"red(手指): cx={xx.mean():.0f} cy={yy.mean():.0f} w={xx.max()-xx.min()} h={yy.max()-yy.min()}")
