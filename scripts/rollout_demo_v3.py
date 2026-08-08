#!/usr/bin/env python3
"""
训练后 SmolVLA 的演示视频生成：随机方块（v3 分布）+ RTC 闭环抓取 + 存帧合成 MP4。
用法（服务器）:
    python rollout_demo_v3.py --ckpt <ckpt> --xml <xml> [--ep 0] [--steps 500]
输出 /tmp/sim_demo_v3/ 下的 front/wrist 帧 + ep.mp4（并排）。
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
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--outdir", default="/tmp/sim_demo_v3")
    args = ap.parse_args()

    import torch
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference.rtc import RTCInferenceEngine
    from lerobot.utils.feature_utils import hw_to_dataset_features
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] loading {args.ckpt}", flush=True)
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
    os.makedirs(args.outdir, exist_ok=True)

    sim.reset()
    R_blk, half_h, face, yaw_deg = sample_block_orientation(rng)
    checker_top = sim.model.geom("checkerboard").pos[2] + sim.model.geom("checkerboard").size[2]
    checker_top_real = checker_top * 1000.0 + T_BASE[2]
    bx = rng.uniform(-120, 30)
    by = rng.uniform(-440, -400)
    bz = checker_top_real + half_h * 1000.0 + 3.0
    sim.set_block_pose_cad(R_BASE @ (np.array([bx, by, bz]) * 1e-3 - T_BASE * 1e-3), mat_to_quat(R_blk))
    for _ in range(120):
        sim.step_delta(np.zeros(7))
    block_z0 = sim.block_pos_real_mm()[2]
    print(f"[demo] block face={face} yaw={yaw_deg}° at ({bx:.0f},{by:.0f},{sim.block_pos_real_mm()[2]:.0f})",
          flush=True)

    engine.start()
    engine.resume()
    max_z = block_z0
    t_start = time.time()
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
                sim.step_delta(act)
                sim.maybe_attach_block(act)
                max_z = max(max_z, sim.block_pos_real_mm()[2])
        if tick % args.save_every == 0:
            Image.fromarray(sim.render("front")).save(f"{args.outdir}/f{tick:04d}_front.png")
            Image.fromarray(sim.render("wrist")).save(f"{args.outdir}/f{tick:04d}_wrist.png")
        elapsed = time.time() - t_tick
        if elapsed < CONTROL_DT:
            time.sleep(CONTROL_DT - elapsed)
        if tick % 100 == 0:
            print(f"[demo] tick {tick} block_z={sim.block_pos_real_mm()[2]:.0f} "
                  f"gripper={st[12]:.0f}% ({time.time()-t_start:.0f}s)", flush=True)
    engine.stop()
    print(f"[demo] block z {block_z0:.0f}->peak {max_z:.0f} (lift {max_z-block_z0:.0f}mm)", flush=True)
    print(f"[demo] frames in {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
