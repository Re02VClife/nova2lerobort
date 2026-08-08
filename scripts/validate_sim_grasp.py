#!/usr/bin/env python3
"""
微调后 SmolVLA 的 RTC 闭环抓取成功率验证
=========================================
随机方块位置跑 N 个 episode，统计：
  - approach：是否接近方块（min dist to block < 50mm）
  - grasp：是否闭爪吸附（attached）
  - carry：是否把方块提起携带（block z 上升 > 20mm）

用法（服务器，nova2_lerobot）:
    conda run -n nova2_lerobot python validate_sim_grasp.py \
        --ckpt <微调 checkpoint> --xml <xml> --episodes 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from sim_smolvla_rollout import (  # noqa: E402
    SimNova2, R_BASE, T_BASE, CONTROL_DT, TASK, build_policy,
)
from record_sim_demos import sample_block_orientation, mat_to_quat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--lerobot-root", default="/home/jjk/桌面/Project/lerobot-main")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference.rtc import RTCInferenceEngine
    from lerobot.utils.feature_utils import hw_to_dataset_features

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[val] loading {args.ckpt}", flush=True)
    policy, pre, post = build_policy(args.ckpt, args.lerobot_root, str(device))

    class _FakeRobot:
        robot_type = "dobot_nova2"
        action_features = {}

    obs_features_hw = {
        "joint_1.pos": float, "joint_2.pos": float, "joint_3.pos": float,
        "joint_4.pos": float, "joint_5.pos": float, "joint_6.pos": float,
        "tcp.x": float, "tcp.y": float, "tcp.z": float,
        "tcp.rx": float, "tcp.ry": float, "tcp.rz": float,
        "gripper.pos": float, "front": (448, 448, 3), "wrist": (640, 480, 3),
    }
    hw_features = hw_to_dataset_features(obs_features_hw, "observation")
    STATE_NAMES = ["joint_1.pos", "joint_2.pos", "joint_3.pos", "joint_4.pos",
                   "joint_5.pos", "joint_6.pos", "tcp.x", "tcp.y", "tcp.z",
                   "tcp.rx", "tcp.ry", "tcp.rz", "gripper.pos"]

    policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    policy.init_rtc_processor()
    engine = RTCInferenceEngine(
        policy, pre, post, _FakeRobot(), policy.config.rtc_config,
        hw_features, TASK, fps=1.0 / CONTROL_DT, device=str(device),
    )

    sim = SimNova2(args.xml)
    rng = np.random.default_rng(args.seed)

    results = []
    t_start = time.time()
    for ep in range(args.episodes):
        sim.reset()
        # 方块姿态随机（与 v3 训练分布一致：朝上的面随机 + yaw 卡迪纳尔）+ 沉降落稳
        R_blk, half_h, face, yaw_deg = sample_block_orientation(rng)
        checker_top = sim.model.geom("checkerboard").pos[2] + sim.model.geom("checkerboard").size[2]
        checker_top_real = checker_top * 1000.0 + T_BASE[2]
        bx = rng.uniform(-120, 30)
        by = rng.uniform(-440, -400)
        bz = checker_top_real + half_h * 1000.0 + 3.0
        sim.set_block_pose_cad(
            R_BASE @ (np.array([bx, by, bz]) * 1e-3 - T_BASE * 1e-3), mat_to_quat(R_blk))
        for _ in range(120):
            sim.step_delta(np.zeros(7))          # 沉降到真实落点
        block_z0 = sim.block_pos_real_mm()[2]

        engine.start()
        engine.resume()
        min_dist = 1e9
        block_z_max = block_z0
        attached_ticks = 0
        for tick in range(args.steps):
            t_tick = time.time()
            st = sim.state_real()
            obs = {n: float(v) for n, v in zip(STATE_NAMES, st)}
            obs["front"] = sim.render("front")
            obs["wrist"] = sim.render("wrist")
            engine.notify_observation(obs)
            action = engine.get_action(None)
            if action is not None:
                act = action.cpu().numpy().astype(np.float64)
                if act.shape == (7,):
                    # 距离：方块中心 vs tcp 的 xy 距离
                    tcp = st[6:9]
                    blk = sim.block_pos_real_mm()
                    d = np.linalg.norm(tcp[:2] - blk[:2])
                    min_dist = min(min_dist, d)
                    sim.step_delta(act)
                    sim.maybe_attach_block(act)
                    if sim.block_attached:
                        attached_ticks += 1
                    block_z_max = max(block_z_max, sim.block_pos_real_mm()[2])
            elapsed = time.time() - t_tick
            if elapsed < CONTROL_DT:
                time.sleep(CONTROL_DT - elapsed)
        engine.stop()

        carried = (block_z_max - block_z0) > 20.0
        approached = min_dist < 50.0
        results.append({"approach": approached, "grasp": attached_ticks > 10, "carry": carried,
                        "min_dist": round(float(min_dist), 1), "lift": round(float(block_z_max - block_z0), 1)})
        r = results[-1]
        print(f"[val] ep {ep+1}: block=({bx:.0f},{by:.0f}) min_dist={r['min_dist']:.0f}mm "
              f"lift={r['lift']:.0f}mm approach={r['approach']} grasp={r['grasp']} carry={r['carry']} "
              f"({time.time()-t_start:.0f}s)", flush=True)

    n_ap = sum(1 for r in results if r["approach"])
    n_gr = sum(1 for r in results if r["grasp"])
    n_ca = sum(1 for r in results if r["carry"])
    print(f"\n[val] {args.episodes} eps: approach {n_ap} ({100*n_ap/args.episodes:.0f}%), "
          f"grasp {n_gr} ({100*n_gr/args.episodes:.0f}%), carry {n_ca} ({100*n_ca/args.episodes:.0f}%)")
    print(f"[val] time {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
