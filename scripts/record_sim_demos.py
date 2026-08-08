#!/usr/bin/env python3
"""
MuJoCo 仿真演示录制：脚本专家抓取黄色方块放到目标区，录成 LeRobot 数据集
==========================================================================
用法（服务器，nova2_lerobot 环境）:
    conda run -n nova2_lerobot python record_sim_demos.py \
        --num-episodes 150 \
        --root ~/dobot_nova2_project/datasets \
        --repo-id local/nova2_sim_grasp

脚本专家策略（IK 直接规划，不靠模型）：
    approach → descend → close(夹爪) → lift → place_hover → place_down → release

每 tick 记录 (front uint8, wrist uint8, state 13, action 7, task)，
用 LeRobot v3.0 录制 API（add_frame / save_episode / finalize），
stats 录制时自动写 meta/stats.json，录完即训。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# 服务器 headless 渲染需要 EGL
os.environ.setdefault("MUJOCO_GL", "egl")

from sim_smolvla_rollout import (  # noqa: E402
    SimNova2, R_BASE, T_BASE, CONTROL_DT, MAX_XYZ_STEP_MM, MAX_ROT_STEP_DEG, TASK,
    zyx_euler_to_rot,
)

# ================================================================
# 方块姿态工具：朝上的面随机（平放/侧立）+ yaw 随机
# ================================================================
def rot_x(deg: float) -> np.ndarray:
    return zyx_euler_to_rot(np.array([deg, 0.0, 0.0]))

def rot_y(deg: float) -> np.ndarray:
    return zyx_euler_to_rot(np.array([0.0, deg, 0.0]))

def rot_z(deg: float) -> np.ndarray:
    return zyx_euler_to_rot(np.array([0.0, 0.0, deg]))

def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> quat [w,x,y,z]（MuJoCo free joint 约定）。"""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j = (i + 1) % 3
        k = (j + 1) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = np.zeros(4)
        q[i + 1] = 0.25 * s
        q[0] = (R[k, j] - R[j, k]) / s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    return q

def sample_block_orientation(rng):
    """随机朝上的面 + 随机 yaw（卡迪纳尔角）。

    方块 60(x)×60(y)×30(z)mm：
      - z 面朝下（平放）→ 高 30mm，up 面 = 60×60
      - x 面朝下（侧立）→ 高 60mm，up 面 = 60×30（local x 竖起来）
      - y 面朝下（侧立）→ 高 60mm，up 面 = 60×30
    yaw 取 {0,90,180,270}°（卡迪纳尔角）：仿真臂在部分位置无法把腕部 yaw 转到任意角
    （实测 (20,-410) 处目标 110° 卡在 -145°），但固定 -90° 夹爪对轴对齐的方块面
    必然垂直，因此限制 yaw 为卡迪纳尔角保证夹爪始终对齐方块面。

    返回 (R_blk_cad, half_h_m, face, yaw_deg)。
    """
    face = rng.choice(["z", "x", "y"])
    yaw_deg = float(rng.choice([0.0, 90.0, 180.0, 270.0]))
    hx, hy, hz = 0.03, 0.03, 0.015
    R_face = np.eye(3)
    half_h = hz
    if face == "x":
        R_face = rot_y(-90.0)   # local x(60mm) -> world z 竖起来
        half_h = hx
    elif face == "y":
        R_face = rot_x(90.0)    # local y(60mm) -> world z
        half_h = hy
    R_blk = rot_z(yaw_deg) @ R_face
    return R_blk, half_h, face, yaw_deg


# ================================================================
# 脚本专家
# ================================================================
class ScriptedExpert:
    def __init__(self, sim, block_real_mm, target_real_mm, rng, half_h_m):
        self.sim = sim
        self.block = np.asarray(block_real_mm, dtype=float)   # 真机 base 系 mm
        self.target = np.asarray(target_real_mm, dtype=float)
        self.rng = rng
        self.target_yaw = -90.0    # 腕部 yaw 固定 -90°：方块 yaw 限卡迪纳尔角，闭合线始终垂直方块面
        bx, by, bz = self.block
        half_h_mm = half_h_m * 1000.0
        block_top = bz + half_h_mm            # 方块顶面 real z（平放72 / 侧立102）
        approach_h = rng.uniform(50.0, 75.0)  # 方块上方高度（含侧立 60mm 块，稳定区 z<=~155）
        lift_h = rng.uniform(60.0, 80.0)      # 抓取后抬升高度
        # tcp 在方块顶上方 2mm，手指垫(约 tcp-20mm)包住方块上部；远工作区不可低于 ~85
        grasp_z = max(block_top + 2.0, 85.0)
        self.phase = "approach"
        self.place_completed = False
        self.max_steps_per_phase = 400
        self.targets = {
            "approach": np.array([bx, by, block_top + approach_h]),
            "descend": np.array([bx, by, grasp_z]),
            "lift": np.array([bx, by, bz + lift_h]),
            "place_hover": np.array([self.target[0], self.target[1], bz + lift_h]),
            "place_down": np.array([self.target[0], self.target[1], 115.0]),  # 高于可达下限；块中心70<盒口100 仍进盒
        }
        self._phase_steps = 0
        self._close_steps = 0

    def _dist(self, cur_xyz):
        tgt = self.targets.get(self.phase)
        return np.linalg.norm(cur_xyz - tgt) if tgt is not None else 0.0

    def step(self):
        """返回 (action7, done)。"""
        self._phase_steps += 1
        cur = self.sim.state_real()[6:12]
        cur_xyz = cur[:3]
        grip = 0.0
        target = None

        if self.phase == "approach":
            target = self.targets["approach"]
            if self._dist(cur_xyz) < 3.0:
                self.phase = "descend"; self._phase_steps = 0
        elif self.phase == "descend":
            target = self.targets["descend"]
            if self._dist(cur_xyz) < 3.0:
                self.phase = "close"; self._phase_steps = 0
        elif self.phase == "close":
            grip = -1.0
            self._close_steps += 1
            # 等夹爪闭到自适应指缝目标（窄块闭得更久，~3s）；超时兜底
            if self.sim.gripper_reached_close() or self._close_steps >= 150:
                self.phase = "lift"; self._phase_steps = 0
        elif self.phase == "lift":
            target = self.targets["lift"]
            if self._dist(cur_xyz) < 3.0:
                self.phase = "place_hover"; self._phase_steps = 0
        elif self.phase == "place_hover":
            target = self.targets["place_hover"]
            if self._dist(cur_xyz) < 3.0:
                self.phase = "place_down"; self._phase_steps = 0
        elif self.phase == "place_down":
            target = self.targets["place_down"]
            grip = -1.0
            if self._dist(cur_xyz) < 3.0:
                self.phase = "release"; self._phase_steps = 0
        elif self.phase == "release":
            grip = 1.0
            self._phase_steps += 1
            if self._phase_steps >= 20:      # 张开约 1 秒
                self.phase = "done"
                self.place_completed = True
                return None, True

        # 相位超时保护
        if self._phase_steps > self.max_steps_per_phase:
            self.phase = "done"
            return None, True

        if target is not None:
            delta = target - cur_xyz
            delta_xyz = np.clip(delta, -MAX_XYZ_STEP_MM, MAX_XYZ_STEP_MM)
        else:
            delta_xyz = np.zeros(3)
        delta_xyz = delta_xyz + self.rng.normal(0, 0.2, 3)   # 微小动作噪声
        # 腕部 yaw 对准：approach/descend 阶段把 tcp rpy_z 转到 target_yaw（-90°+方块yaw），
        # 使手指闭合线始终垂直于方块面；用 wrap 到 [-180,180] 取最短旋转路径。
        cur_rpy = cur[3:6]
        delta_rot = np.zeros(3)
        if self.phase in ("approach", "descend"):
            d = self.target_yaw - cur_rpy[2]
            d = (d + 180.0) % 360.0 - 180.0
            delta_rot[2] = np.clip(d, -MAX_ROT_STEP_DEG, MAX_ROT_STEP_DEG)
        return np.concatenate([delta_xyz, delta_rot, [grip]]).astype(np.float64), False


# ================================================================
# 数据集 features（复用真实 nova2 数据集结构）
# ================================================================
def make_features():
    return {
        "observation.images.front": {
            "dtype": "video", "shape": [448, 448, 3], "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        },
        "observation.images.wrist": {
            "dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        },
        "observation.state": {
            "dtype": "float32",
            "names": ["joint_1.pos", "joint_2.pos", "joint_3.pos", "joint_4.pos",
                      "joint_5.pos", "joint_6.pos", "tcp.x", "tcp.y", "tcp.z",
                      "tcp.rx", "tcp.ry", "tcp.rz", "gripper.pos"],
            "shape": [13],
        },
        "action": {
            "dtype": "float32",
            "names": ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"],
            "shape": [7],
        },
    }


def _block_yellow_mask(img):
    """黄色方块像素掩码（front 448x448 / wrist 640x480 通用）。"""
    return ((img[..., 0] > 150) & (img[..., 1] > 100) & (img[..., 1] < 200)
            & (img[..., 2] < 120) & (img[..., 0] - img[..., 1] > 30))


def run_test(sim, target_real):
    """不录数据，验证姿态随机采集链路：
    1) 位置往返一致  2) 各朝上的面(z/x/y)+yaw 放置稳定(不倾倒)  3) 完整抓取闭环
    4) front/wrist 相机能看到方块（黄色像素）。"""
    from PIL import Image
    rng = np.random.default_rng(123)
    outdir = "/tmp/sim_record_test"
    os.makedirs(outdir, exist_ok=True)
    print(f"[test] 目标区(黑盒内) real mm: {np.round(target_real, 1).tolist()}", flush=True)

    # 棋盘格顶面 real z（用于方块落位）
    checker_top = sim.model.geom("checkerboard").pos[2] + sim.model.geom("checkerboard").size[2]
    checker_top_real = checker_top * 1000.0 + T_BASE[2]
    print(f"[test] checkerboard top real z = {checker_top_real:.1f}mm", flush=True)

    # 1) 位置往返一致性（棋盘格内可达区，平放）
    for i, pt in enumerate([(0.0, -420.0, 57.0), (-100.0, -430.0, 57.0),
                            (30.0, -410.0, 57.0), (-120.0, -440.0, 57.0), (20.0, -400.0, 57.0)]):
        sim.reset()
        sim.set_block_pos_cad(R_BASE @ (np.array(pt) * 1e-3 - T_BASE * 1e-3))
        got = sim.block_pos_real_mm()
        err = np.linalg.norm(got - np.array(pt))
        print(f"[test] pos#{i} tgt={pt} got={np.round(got, 1).tolist()} err={err:.1f}mm", flush=True)

    # 2) 姿态放置稳定性：各面 + 卡迪纳尔 yaw，放好(微离面3mm)沉降 120 步看方块是否稳定
    print("[test] --- 姿态沉降（放好→沉降120步→落稳位置/半高） ---", flush=True)
    for i, (face, yaw) in enumerate([("z", 0.0), ("z", 90.0), ("x", 0.0), ("x", 180.0),
                                     ("y", 90.0), ("y", 270.0)]):
        sim.reset()
        hx, hy, hz = 0.03, 0.03, 0.015
        Rf = {"z": np.eye(3), "x": rot_y(-90.0), "y": rot_x(90.0)}[face]
        half_h = {"z": hz, "x": hx, "y": hy}[face]
        R_blk = rot_z(yaw) @ Rf
        bz = checker_top_real + half_h * 1000.0 + 3.0
        block_real = np.array([0.0, -420.0, bz])
        block_cad = R_BASE @ (block_real * 1e-3 - T_BASE * 1e-3)
        sim.set_block_pose_cad(block_cad, mat_to_quat(R_blk))
        for _ in range(120):
            sim.step_delta(np.zeros(7))
        b1 = sim.block_pos_real_mm()
        xm = sim.block_xmat()
        z_extent = np.abs(xm[2, :]) @ np.array([hx, hy, hz])
        ok = abs(z_extent - half_h) < 0.003
        print(f"[test] face={face} yaw={yaw:g}°: settled=({b1[0]:.0f},{b1[1]:.0f},{b1[2]:.1f}) "
              f"z_extent={z_extent*1000:.0f}mm expect={half_h*1000:.0f}mm ok={ok}", flush=True)

    # 3) 完整抓取闭环（各姿态：平放/侧立x/侧立y × 卡迪纳尔 yaw）
    print("[test] --- 完整抓取闭环 ---", flush=True)
    for i, (face, yaw, pt) in enumerate([("z", 90.0, (0.0, -420.0, 57.0)),
                                         ("x", 0.0, (-80.0, -425.0, 72.0)),
                                         ("y", 180.0, (20.0, -410.0, 72.0)),
                                         ("z", 270.0, (-100.0, -430.0, 57.0)),
                                         ("x", 90.0, (0.0, -415.0, 72.0))]):
        sim.reset()
        hx, hy, hz = 0.03, 0.03, 0.015
        Rf = {"z": np.eye(3), "x": rot_y(-90.0), "y": rot_x(90.0)}[face]
        half_h = {"z": hz, "x": hx, "y": hy}[face]
        R_blk = rot_z(yaw) @ Rf
        block_real = np.array(pt, dtype=float) + np.array([0.0, 0.0, 3.0])
        block_cad = R_BASE @ (block_real * 1e-3 - T_BASE * 1e-3)
        sim.set_block_pose_cad(block_cad, mat_to_quat(R_blk))
        for _ in range(120):
            sim.step_delta(np.zeros(7))          # 沉降到落稳位置
        block_real = sim.block_pos_real_mm()
        R_settled = sim.block_xmat()
        half_h = float(np.abs(R_settled[2, :]) @ np.array([hx, hy, hz]))
        expert = ScriptedExpert(sim, block_real, target_real, rng, half_h)
        block_z0 = sim.block_pos_real_mm()[2]
        max_block_z = block_z0
        z_before_close = block_z0
        max_steps = 700
        for t in range(max_steps):
            action7, done = expert.step()
            if done:
                break
            if expert.phase == "close" and z_before_close == block_z0:
                z_before_close = sim.block_pos_real_mm()[2]
            if expert.phase == "close" and t < 120:
                front = sim.render("front")
                wrist = sim.render("wrist")
                Image.fromarray(front).save(f"{outdir}/ep{i}_t{t:03d}_front.png")
                Image.fromarray(wrist).save(f"{outdir}/ep{i}_t{t:03d}_wrist.png")
                blk_px = int(_block_yellow_mask(front).sum())
                wb_px = int(_block_yellow_mask(wrist).sum())
                if t % 10 == 0:
                    print(f"[test] ep{i} face={face} yaw={yaw:g}° close: "
                          f"front_px={blk_px} wrist_px={wb_px}", flush=True)
            sim.step_delta(action7)
            sim.maybe_attach_block(action7)
            max_block_z = max(max_block_z, sim.block_pos_real_mm()[2])
        z1 = sim.block_pos_real_mm()[2]
        stable = abs(z_before_close - block_z0) < 3.0
        lifted = (max_block_z - block_z0) > 15.0
        print(f"[test] ep{i} face={face} yaw={yaw:g}°: block z {block_z0:.0f}->peak {max_block_z:.0f}"
              f"(end {z1:.0f}) stable={stable} lifted={lifted} placed={expert.place_completed} "
              f"phase={expert.phase}", flush=True)
    print(f"[test] 帧已存 {outdir}/ 供检查（front/wrist）", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="/home/jjk/dobot_nova2_project/sim/nova2_scene_corrected.xml")
    ap.add_argument("--root", default="/home/jjk/dobot_nova2_project/datasets")
    ap.add_argument("--repo-id", default="local/nova2_sim_grasp")
    ap.add_argument("--num-episodes", type=int, default=150)
    ap.add_argument("--max-steps", type=int, default=700, help="单 episode 最大步数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test", action="store_true", help="只跑验证，不录数据")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sim = SimNova2(args.xml)
    sim.reset()

    # 目标区 = 黑色盒子内部：盒中心 real x≈154 臂不可达，用盒内可靠放置点 x=70
    box_cad = sim.model.body("black_box").pos.copy()
    box_real = R_BASE @ (box_cad * 1000.0) + T_BASE
    target_real = np.array([70.0, box_real[1], 55.0])
    print(f"[rec] 目标区(黑盒内) real mm: {np.round(target_real, 1).tolist()}", flush=True)
    if args.test:
        run_test(sim, target_real)
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = os.path.join(args.root, args.repo_id.split("/")[-1])
    ds = LeRobotDataset.create(
        args.repo_id, fps=20, features=make_features(), root=dataset_root,
        robot_type="dobot_nova2", use_videos=True, streaming_encoding=True,
    )

    success = 0
    attempts = 0
    t_start = time.time()
    # 棋盘格顶面 real z（方块落位基准）
    checker_top = sim.model.geom("checkerboard").pos[2] + sim.model.geom("checkerboard").size[2]
    checker_top_real = checker_top * 1000.0 + T_BASE[2]
    while success < args.num_episodes and attempts < args.num_episodes * 3:
        attempts += 1
        sim.reset()
        # 方块姿态：朝上的面随机(z平放30mm / x,y侧立60mm) + yaw 随机卡迪纳尔角 {0,90,180,270}°
        R_blk, half_h, face, yaw_deg = sample_block_orientation(rng)
        # 方块出现位置 = 真机 base 系 mm：随机生成在棋盘格内（棋盘格 x∈[-326,34] y∈[-633,-273]，
        # 取仿真臂可达交集 x∈[-120,30] y∈[-440,-400]，z85 下探验证区间）。
        bx = rng.uniform(-120, 30)
        by = rng.uniform(-440, -400)
        # 方块中心 z = 棋盘格顶 + 半高 + 3mm 离面高度（微落，避免零穿透接触的小倾覆）
        bz = checker_top_real + half_h * 1000.0 + 3.0
        block_real = np.array([bx, by, bz])
        block_cad = R_BASE @ (block_real * 1e-3 - T_BASE * 1e-3)
        sim.set_block_pose_cad(block_cad, mat_to_quat(R_blk))
        # 沉降：方块在重力+接触下落到真实稳定落点（手臂停在 home），再按落稳位置规划
        for _ in range(120):
            sim.step_delta(np.zeros(7))
        block_real = sim.block_pos_real_mm()                     # 落稳后位置
        R_settled = sim.block_xmat()
        half_h = float(np.abs(R_settled[2, :]) @ np.array([0.03, 0.03, 0.015]))  # 实际半高
        # 落稳后半高须接近合法值（平放15mm / 侧立30mm），否则方块卡棱/翻倒 → 重采样本条
        if not (abs(half_h - 0.015) < 0.003 or abs(half_h - 0.030) < 0.003):
            print(f"[rec] attempt {attempts}: block 落稳异常 half_h={half_h*1000:.0f}mm，重采样", flush=True)
            continue

        expert = ScriptedExpert(sim, block_real, target_real, rng, half_h)
        block_z0 = sim.block_pos_real_mm()[2]
        max_block_z = block_z0
        z_before_close = block_z0
        block_stable = True

        # 运动学吸附抓取：夹爪闭合后把方块刚体吸附到 TCP（sim.maybe_attach_block）
        n_frames = 0
        done = False
        for t in range(args.max_steps):
            action7, done = expert.step()
            if done:
                break

            # 抓取：进入 close 相位时检查方块是否仍稳定（被撞开/倾倒则放弃本条）
            if expert.phase == "close":
                if z_before_close == block_z0:   # 只记第一次
                    z_before_close = sim.block_pos_real_mm()[2]
                if abs(z_before_close - block_z0) > 3.0:
                    block_stable = False

            front = sim.render("front")
            wrist = sim.render("wrist")
            state = sim.state_real()
            frame = {
                "observation.images.front": front,
                "observation.images.wrist": wrist,
                "observation.state": state,
                "action": action7.astype(np.float32),
                "task": TASK,
            }
            ds.add_frame(frame)
            sim.step_delta(action7)
            sim.maybe_attach_block(action7)      # 闭爪吸附 / 张开释放（保持方块姿态）
            n_frames += 1
            max_block_z = max(max_block_z, sim.block_pos_real_mm()[2])

        # 成功判据：方块在闭合前未移位 且 被真正提起过 且 完整走完放置+释放
        lifted = (max_block_z - block_z0) > 15.0
        if block_stable and lifted and expert.place_completed:
            success += 1
            ds.save_episode()
        else:
            ds.clear_episode_buffer()   # 失败/未完成放置不入数据集

        if success % 10 == 0 or success == args.num_episodes:
            print(f"[rec] {success}/{args.num_episodes} eps (attempt {attempts}): "
                  f"{n_frames} frames, face={face} yaw={yaw_deg:.0f}° "
                  f"block z {block_z0:.0f}->peak {max_block_z:.0f}mm "
                  f"stable={block_stable} ({time.time()-t_start:.0f}s)", flush=True)

    ds.finalize()
    print(f"\n[rec] DONE {success} eps in {attempts} attempts "
          f"({100*success/max(1,attempts):.0f}% grasp rate), {time.time()-t_start:.0f}s")
    print(f"[rec] dataset: {dataset_root}")


if __name__ == "__main__":
    main()
