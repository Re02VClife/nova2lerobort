#!/usr/bin/env python3
"""
在真机 ep0 hover 姿态下，用多组 front 相机参数渲染，保存 PNG 供对比。
相机 zaxis 约定：相机沿 -zaxis 看（MuJoCo），zaxis=(0, -sin, +cos) = 向前(+y)下方看。
"""
import os, re
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from PIL import Image
from sim_smolvla_rollout import (SimNova2, R_BASE, T_BASE, GRIPPER_OPEN_M,
                                 zyx_euler_to_rot)

XML = "/home/jjk/dobot_nova2_project/sim/nova2_scene_corrected.xml"
xml = open(XML).read()
xml = xml.replace('mat_block" rgba="0.85 0.65 0.33', 'mat_block" rgba="1.0 0.0 1.0')

block_real = np.array([-112.0, -379.0, 70.0])
block_cad = R_BASE @ (block_real * 1e-3 - T_BASE * 1e-3)
tcp_real = np.array([-112.0, -379.0, 64.2])
rpy = np.array([-179.7, 2.6, 165.8])
tcp_cad = R_BASE @ (tcp_real * 1e-3 - T_BASE * 1e-3)
tcp_mat = R_BASE @ zyx_euler_to_rot(rpy)

variants = [
    ("v_orig", '<camera name="front" pos="0.004 0.367 0.666" zaxis="0 0 1" fovy="40" resolution="448 448" />'),
    ("v_shift_x", '<camera name="front" pos="-0.06 0.367 0.666" zaxis="0 0 1" fovy="40" resolution="448 448" />'),
    ("v_tilt20", '<camera name="front" pos="0 0.40 0.78" zaxis="0 -0.3420 0.9397" fovy="50" resolution="448 448" />'),
    ("v_tilt30", '<camera name="front" pos="0 0.40 0.80" zaxis="0 -0.5000 0.8660" fovy="55" resolution="448 448" />'),
    ("v_tilt40", '<camera name="front" pos="0 0.40 0.85" zaxis="0 -0.6428 0.7660" fovy="60" resolution="448 448" />'),
    ("v_tilt25_low", '<camera name="front" pos="0 0.44 0.70" zaxis="0 -0.4226 0.9063" fovy="55" resolution="448 448" />'),
]

for name, cam in variants:
    xml2 = re.sub(r'<camera name="front"[^>]*/>', cam, xml)
    tmp = XML + f".{name}.xml"
    open(tmp, "w").write(xml2)
    s = SimNova2(tmp)
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
    Image.fromarray(img).save(f"/tmp/fr_{name}.png")
    lum = img.mean(axis=2)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mg = (r > 70) & (b > 70) & (g < 45)
    info = ""
    if mg.sum() > 5:
        yy, xx = np.where(mg)
        info = f" block@({int(xx.mean())},{int(yy.mean())}) w={int(xx.max()-xx.min())}"
    yel = (r > 200) & (g > 160) & (b < 120) & (r - g > 30)
    if yel.sum() > 20:
        yy, xx = np.where(yel)
        info += f" pad@({int(xx.mean())},{int(yy.mean())})"
    print(f"{name}: lum={lum.mean():.0f} nonblack={(lum>5).mean()*100:.0f}%{info}", flush=True)
