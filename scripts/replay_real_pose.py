#!/usr/bin/env python3
"""
把真机数据集某帧的 TCP 位姿重放到仿真，渲染 front/wrist 相机，
对比方块/夹爪像素位置，用于顶视相机标定。
用法（服务器 EGL）:
    conda run -n nova2_lerobot python replay_real_pose.py \
        --tcp /tmp/real_grasp/_tcp.json --out /tmp/replay
"""
from __future__ import annotations
import argparse, json, os

import numpy as np
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")

from sim_smolvla_rollout import (SimNova2, R_BASE, T_BASE, GRIPPER_OPEN_M,
                                 rot_to_zyx_euler, zyx_euler_to_rot)


def magenta_detect(img):
    r, g, b = img[..., 0].astype(float), img[..., 1].astype(float), img[..., 2].astype(float)
    m = (r > 70) & (b > 70) & (g < 45)
    if m.sum() < 40:
        return None
    ys, xs = np.where(m)
    return {
        "cx": float(xs.mean()), "cy": float(ys.mean()),
        "w": float(xs.max() - xs.min()), "h": float(ys.max() - ys.min()),
        "px": int(m.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="/home/jjk/dobot_nova2_project/sim/nova2_scene_corrected.xml")
    ap.add_argument("--tcp", default="/tmp/real_grasp/_tcp.json")
    ap.add_argument("--out", default="/tmp/replay")
    args = ap.parse_args()

    import mujoco
    os.makedirs(args.out, exist_ok=True)

    xml = open(args.xml).read()
    xml = xml.replace('mat_block" rgba="0.85 0.65 0.33', 'mat_block" rgba="1.0 0.0 1.0')
    tmp = args.xml + ".replay_tmp.xml"
    open(tmp, "w").write(xml)

    sim = SimNova2(tmp)
    sim.reset()
    mujoco.mj_forward(sim.model, sim.data)
    fq = sim.model.joint("finger_left_joint").qposadr[0]
    sim.data.qpos[fq] = GRIPPER_OPEN_M
    sim.data.qpos[fq + 1] = GRIPPER_OPEN_M

    data = json.load(open(args.tcp))
    results = {}
    for ep_key in sorted(data):
        m = data[ep_key]
        tcp_r = m["hover"]["tcp"]
        # 方块：真机在爪正下方（tcp 的 x/y），z 固定桌面上方（CAD 桌顶 -0.094 + 30mm）
        block_real = np.array([tcp_r[0], tcp_r[1], 70.0])
        block_cad = R_BASE @ (block_real * 1e-3 - T_BASE * 1e-3)
        sim.set_block_pos_cad(block_cad)

        # 手臂：IK 到真机 TCP 位姿（z 用 tcp z 直接换算）
        tcp_pos_real = np.array(tcp_r[:3])
        tcp_rpy_deg = np.array([0.0, 0.0, 0.0])  # 姿态用 hover 的 rpy
        hover = m["hover"]
        if "rpy" in hover:
            tcp_rpy_deg = np.array(hover["rpy"])
        tcp_pos_cad = R_BASE @ (tcp_pos_real * 1e-3 - T_BASE * 1e-3)
        tcp_mat_cad = R_BASE @ zyx_euler_to_rot(tcp_rpy_deg)
        q = sim.ik_to_q(tcp_pos_cad, tcp_mat_cad)
        sim.data.qpos[:6] = q
        mujoco.mj_forward(sim.model, sim.data)

        entry = {"block_real": [round(v, 1) for v in block_real],
                 "block_cad": [round(v, 4) for v in block_cad],
                 "tcp_cad": [round(v, 4) for v in tcp_pos_cad]}
        for cam in ("front", "wrist"):
            img = sim.render(cam)
            Image.fromarray(img).save(os.path.join(args.out, f"replay_{ep_key}_{cam}.png"))
            b = magenta_detect(img)
            entry[cam] = b
            print(f"{ep_key} {cam} block_px={b}", flush=True)
        results[ep_key] = entry
        print(f"{ep_key} tcp_cad={tcp_pos_cad.round(4)}", flush=True)

    with open(os.path.join(args.out, "_replay.json"), "w") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("[replay] DONE ->", args.out)


if __name__ == "__main__":
    main()
