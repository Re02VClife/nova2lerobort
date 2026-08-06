#!/usr/bin/env python3
"""
SmolVLA 闭环 MuJoCo 仿真 rollout
=================================
在服务器上跑：把训练好的 SmolVLA(210eps) 接进 MuJoCo，机器人按模型输出的动作
在仿真里闭环执行抓取。完全复用 LeRobot 真机 rollout 的推理链：

    obs_frame → prepare_observation_for_inference → preprocessor(归一化)
              → policy.select_action(flow matching 分块) → postprocessor(反归一化)
              → 7 维 delta 动作 [dx,dy,dz](mm)+[drx,dry,drz](°)+grip

CAD 坐标系与真机 base 坐标系的标定映射（由 home/SAFE_A 两姿态反推）：
    real = R_BASE * cad + T        (位置)
    realR = R_BASE * cadR          (姿态)
    q_cad = SIGN * q_real          (关节方向)
    SIGN=[1,-1,1,-1,1,1]  R_BASE=diag(-1,-1,1)  T=[3.7,17.1,133.9]mm

动作执行：target_pose = current_tcp + delta（基座系、欧拉角逐元素相加，镜像真机）
         → Jacobian DLS IK → 6 关节目标 → 显式 PD 力矩伺服（保留 motor 执行器）。

用法（服务器）:
    conda run -n nova2_lerobot python sim_smolvla_rollout.py \
        --ckpt ~/dobot_nova2_project/outputs/train/nova2_smolvla_multitask_210eps/checkpoints/last/pretrained_model \
        --xml ~/dobot_nova2_project/sim/nova2_scene_corrected.xml \
        --steps 2000 --log /tmp/sim_rollout.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# 服务器 headless 渲染需要 EGL
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import torch  # noqa: E402

# ================================================================
# 标定常量（CAD 系 <-> 真机 base 系）
# ================================================================
SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 1.0])       # q_cad = SIGN * q_real
R_BASE = np.diag([-1.0, -1.0, 1.0])                     # real = R_BASE*cad + T
T_BASE = np.array([3.7, 17.1, 133.9])                   # mm

# 控制 / 场景常量
CONTROL_DT = 0.05          # 一个动作步对应 0.05s（20Hz，对齐数据集 fps）
MAX_XYZ_STEP_MM = 4.0      # 单步平移限幅（镜像真机 clamp_delta_action）
MAX_ROT_STEP_DEG = 2.0     # 单步旋转限幅
GRIPPER_THRESHOLD = 0.5    # 夹爪动作阈值
GRIPPER_OPEN_M = 0.0721    # 手指全开行程：横向60mm(垫片±50mm=100mm开口)+后缩40mm，闭合时向前延申4cm(真机特性)
GRIP_KP = 500.0            # 夹爪 PD 刚度（N/m）
GRIP_KD = 2.0 * np.sqrt(GRIP_KP * 0.08)  # 手指质量 0.08kg，临界阻尼
GRIP_FRC = 50.0            # 夹爪力矩限幅
IK_LAMBDA = 0.01           # DLS 阻尼
IK_ITERS = 3
# 自适应临界阻尼 PD：kp = OMEGA²·I_eff, kd = 2ζ·sqrt(kp·I_eff)，I_eff 取 qM 对角（真实惯性）
OMEGA = 120.0   # rad/s 目标带宽 -> ~37ms 收敛，ω·dt=0.24 < 0.3 稳定
ZETA = 0.9
# 关节力矩限幅（XML 里在 joint 上用 actuatorfrcrange 声明，motor 执行器本身不限幅）
FRC_LIMITS = np.array([30.0, 30.0, 30.0, 10.0, 10.0, 5.0])
POSE_LIMITS_MM = np.array([[-380, 380], [-560, -150], [0, 300],
                           [-360, 360], [-360, 360], [-360, 360]])

TASK = "抓取黄色方块并放到目标区域"

# 相机后处理（对标真机相机特性）
WRIST_WB_ENABLE = True    # 真机腕部相机自动曝光/白平衡（保留红色夹爪，中和偏暖背景）
WRIST_WB_SAT = 0.85       # 保留 85% 饱和度（真实腕部图红色夹爪清晰，未强降饱和）
WRIST_WB_GAMMA = 1.0      # 白平衡不压亮度，亮度由光照决定
WRIST_WB_GAIN = 1.55      # 真机腕部相机自动曝光：曝光增益（夹爪100mm+fovy49 后画面亮，增益平衡亮度）
WRIST_WB_CONTRAST = 1.50  # 对比拉伸（中心 128）：高光裁切、阴影压黑，模拟相机高动态响应


def wrist_camera_wb(img: np.ndarray) -> np.ndarray:
    """真机腕部相机白平衡/自动曝光后处理：中度降饱和 + 曝光增益 + 对比拉伸（高光裁切）。"""
    arr = img.astype(np.float32)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    out = arr * WRIST_WB_SAT + lum[..., None] * (1.0 - WRIST_WB_SAT)
    out = (out / 255.0) ** WRIST_WB_GAMMA * 255.0
    out = out * WRIST_WB_GAIN
    out = 128.0 + (out - 128.0) * WRIST_WB_CONTRAST
    return np.clip(out, 0, 255).astype(np.uint8)


# ================================================================
# 旋转工具
# ================================================================
def rot_to_zyx_euler(R: np.ndarray) -> np.ndarray:
    """R = Rz(y)*Ry(p)*Rx(r)，返回 [r,p,y] 度。"""
    r = np.arctan2(R[2, 1], R[2, 2])
    s = np.hypot(R[0, 0], R[1, 0])
    p = np.arctan2(-R[2, 0], s)
    y = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([r, p, y])


def zyx_euler_to_rot(rpy_deg: np.ndarray) -> np.ndarray:
    """zyx_euler_to_rot 的逆。"""
    r, p, y = np.radians(rpy_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> 轴角旋转向量（3,）。"""
    R = np.asarray(R, dtype=float)
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * np.sin(angle)) * angle


# ================================================================
# MuJoCo 仿真机器人
# ================================================================
class SimNova2:
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.tcp_id = self.model.site("tcp").id
        self.gripper_joint_id = self.model.joint("finger_left_joint").id
        self.gripper_act_id = self.model.actuator("act_gripper").id
        self.gripper_act_r_id = self.model.actuator("act_gripper_r").id
        self.act_ids = [self.model.actuator(f"act_J{i}").id for i in range(1, 7)]
        # 黄色方块 free joint（worldbody 末尾 → qpos 索引 8 起）
        self.block_jnt = self.model.body("block").jntadr[0]
        self.block_qposadr = self.model.jnt_qposadr[self.block_jnt]
        self.block_dofadr = self.model.jnt_dofadr[self.block_jnt]
        self._Mfull = np.zeros((self.model.nv, self.model.nv))  # 供 mj_fullM 展开稀疏质量矩阵
        # 每个相机的离屏渲染器
        self.renderers = {}
        for cam in ("front", "wrist"):
            cid = self.model.camera(cam).id
            w, h = self.model.cam_resolution[cid]
            self.renderers[cam] = mujoco.Renderer(self.model, int(h), int(w))
        # PD 目标初始化为 home
        self.q_target = None
        self.grip_target = GRIPPER_OPEN_M
        self.default_block_pos_cad = np.array([0.004, 0.437, -0.079])  # 桌面上的默认位置(工作台36x66x8,机械臂远离10cm)
        # 运动学吸附（与训练数据一致）：闭爪且方块在 TCP 附近时吸附
        self.block_attached = False
        self.grasp_offset_cad = None

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.ctrl[:] = 0.0
        self.grip_target = GRIPPER_OPEN_M  # 夹爪初始张开
        self.set_block_pos_cad(self.default_block_pos_cad)  # 方块放回桌面默认位置
        self.block_attached = False
        self.grasp_offset_cad = None
        mujoco.mj_forward(self.model, self.data)
        self.q_target = self.data.qpos[:6].copy()

    def home_joints_cad(self) -> np.ndarray:
        """返回 CAD 系 home qpos（前 6 维）。"""
        self.reset()
        return self.data.qpos[:6].copy()

    def tcp_cad(self) -> tuple[np.ndarray, np.ndarray]:
        """CAD 系 TCP 位置(m) + 旋转矩阵。"""
        pos = self.data.site_xpos[self.tcp_id].copy()
        mat = self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()
        return pos, mat

    def gripper_percent(self) -> float:
        q = self.data.qpos[self.model.joint("finger_left_joint").qposadr[0]]
        return float(np.clip(q / GRIPPER_OPEN_M, 0, 1) * 100.0)

    def state_real(self) -> np.ndarray:
        """13 维真机观测：[6 关节角(°), 6 TCP(mm/°), 夹爪(0-100)]。"""
        q_cad = self.data.qpos[:6]
        joints_deg = np.degrees(SIGN * q_cad)
        pos_cad, mat_cad = self.tcp_cad()
        pos_real_mm = R_BASE @ (pos_cad * 1000.0) + T_BASE
        mat_real = R_BASE @ mat_cad
        rpy_deg = rot_to_zyx_euler(mat_real)
        grip = self.gripper_percent()
        return np.concatenate([joints_deg, pos_real_mm, rpy_deg, [grip]]).astype(np.float32)

    def render(self, cam: str) -> np.ndarray:
        r = self.renderers[cam]
        r.update_scene(self.data, camera=cam)
        img = r.render()  # uint8 HWC RGB
        if cam == "wrist" and WRIST_WB_ENABLE:
            img = wrist_camera_wb(img)
        return img

    # ---- 黄色方块（free body）----
    def set_block_pos_cad(self, pos_cad: np.ndarray):
        bq = self.block_qposadr
        self.data.qpos[bq:bq + 3] = np.asarray(pos_cad, dtype=float)
        self.data.qpos[bq + 3:bq + 7] = np.array([1.0, 0, 0, 0])
        bd = self.block_dofadr
        self.data.qvel[bd:bd + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def block_pos_cad(self) -> np.ndarray:
        return self.data.body("block").xpos.copy()

    def block_pos_real_mm(self) -> np.ndarray:
        return R_BASE @ (self.block_pos_cad() * 1000.0) + T_BASE

    def maybe_attach_block(self, action7: np.ndarray):
        """与训练一致的运动学吸附：模型命令闭爪且方块在 TCP 附近时，方块跟随 TCP。"""
        grip = float(action7[6])
        if grip <= -GRIPPER_THRESHOLD and not self.block_attached:
            block_cad = self.block_pos_cad()
            tcp_cad = self.tcp_cad()[0]
            if np.linalg.norm(block_cad - tcp_cad) < 0.08:   # 8cm 内视为抓住
                self.grasp_offset_cad = block_cad - tcp_cad
                self.block_attached = True
        elif grip >= GRIPPER_THRESHOLD and self.block_attached:
            self.block_attached = False
        if self.block_attached and self.grasp_offset_cad is not None:
            self.set_block_pos_cad(self.tcp_cad()[0] + self.grasp_offset_cad)

    # ---- IK ----
    def jacobian(self) -> np.ndarray:
        """TCP 站点 6x6 雅可比（位置+旋转），只取 6 个臂关节。"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp_id)
        J = np.vstack([jacp[:, :6], jacr[:, :6]])  # (6,6)
        return J

    def ik_to_q(self, target_pos_cad: np.ndarray, target_mat_cad: np.ndarray) -> np.ndarray:
        """Jacobian DLS IK：从当前 q 迭代求 CAD qpos 使 TCP 达到目标位姿。"""
        q = self.data.qpos[:6].copy()
        lam2 = IK_LAMBDA ** 2
        for _ in range(IK_ITERS):
            self.data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.data)
            pos, mat = self.tcp_cad()
            e_pos = target_pos_cad - pos
            e_rot = rot_to_axis_angle(target_mat_cad @ mat.T)
            e = np.concatenate([e_pos, e_rot])
            J = self.jacobian()
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
            q = q + dq
        return q

    # ---- 执行 ----
    def step_delta(self, action7: np.ndarray):
        """执行一个 7 维 delta 动作（真机 base 系）。"""
        delta_mm = np.clip(action7[:3], -MAX_XYZ_STEP_MM, MAX_XYZ_STEP_MM) * 1e-3
        delta_deg = np.clip(action7[3:6], -MAX_ROT_STEP_DEG, MAX_ROT_STEP_DEG)

        # 真机语义：target_pose = current_tcp + delta（欧拉角逐元素相加，base 系）
        pos_cad, mat_cad = self.tcp_cad()
        mat_real = R_BASE @ mat_cad
        real_rpy = rot_to_zyx_euler(mat_real)
        target_real_rpy = real_rpy + delta_deg
        target_real_pos_mm = (R_BASE @ (pos_cad * 1000.0)) + T_BASE
        target_real_pos_mm = target_real_pos_mm + delta_mm * 1000.0  # delta 累加（镜像 move_delta）

        # 位姿限幅
        target_real_pos_mm = np.clip(target_real_pos_mm, POSE_LIMITS_MM[:3, 0], POSE_LIMITS_MM[:3, 1])
        target_real_rpy = np.clip(target_real_rpy, POSE_LIMITS_MM[3:, 0], POSE_LIMITS_MM[3:, 1])

        # 转回 CAD 系
        target_pos_cad = R_BASE @ (target_real_pos_mm * 1e-3 - T_BASE * 1e-3)
        target_mat_cad = R_BASE @ zyx_euler_to_rot(target_real_rpy)

        self.q_target = self.ik_to_q(target_pos_cad, target_mat_cad)

        # 夹爪：阈值二值化 -> 更新 grip_target（开/合/保持）
        grip = float(action7[6])
        if abs(grip) >= GRIPPER_THRESHOLD:
            self.grip_target = GRIPPER_OPEN_M if grip > 0 else 0.0

        # 自适应临界阻尼 PD（手臂）+ 夹爪 PD + 重力补偿，力矩伺服 CONTROL_DT 时长
        n_sub = max(1, int(round(CONTROL_DT / self.model.opt.timestep)))
        for _ in range(n_sub):
            q = self.data.qpos[:6]
            qv = self.data.qvel[:6]
            mujoco.mj_fullM(self.model, self.data, self._Mfull)
            I_eff = np.diag(self._Mfull)[:6].copy()
            kp = OMEGA ** 2 * I_eff
            kd = 2.0 * ZETA * np.sqrt(kp * I_eff)
            bias = self.data.qfrc_bias[:6].copy()  # 重力/离心/科氏补偿
            tau = kp * (self.q_target - q) - kd * qv + bias
            tau = np.clip(tau, -FRC_LIMITS, FRC_LIMITS)
            for i, aid in enumerate(self.act_ids):
                self.data.ctrl[aid] = tau[i]
            # 夹爪：左右指同目标位置 PD（镜像）
            for gid, gq, gqvel in ((self.gripper_act_id, self.data.qpos[6], self.data.qvel[6]),
                                   (self.gripper_act_r_id, self.data.qpos[7], self.data.qvel[7])):
                self.data.ctrl[gid] = np.clip(
                    GRIP_KP * (self.grip_target - gq) - GRIP_KD * gqvel, -GRIP_FRC, GRIP_FRC)
            mujoco.mj_step(self.model, self.data)


# ================================================================
# 主流程
# ================================================================
def build_policy(ckpt: str, lerobot_root: str, device: str):
    sys.path.insert(0, os.path.join(lerobot_root, "src"))
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.vlm_model_name = os.path.join(lerobot_root, "models/SmolVLM2-500M-Video-Instruct")
    policy = SmolVLAPolicy.from_pretrained(ckpt, config=cfg)
    policy = policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt)
    return policy, pre, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SmolVLA checkpoint (pretrained_model 目录)")
    ap.add_argument("--xml", required=True, help="MuJoCo XML")
    ap.add_argument("--lerobot-root", default="/home/jjk/桌面/Project/lerobot-main")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--log", default="/tmp/sim_rollout.jsonl")
    ap.add_argument("--save-frames-every", type=int, default=20, help="每 N tick 存一帧到 /tmp/sim_frames")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--inference", choices=["sync", "rtc"], default="rtc",
                    help="推理模式：sync=select_action 同步队列；rtc=异步分块(RTCInferenceEngine)")
    args = ap.parse_args()

    from lerobot.policies.utils import prepare_observation_for_inference

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[sim] loading policy from {args.ckpt} ...", flush=True)
    policy, pre, post = build_policy(args.ckpt, args.lerobot_root, str(device))

    print(f"[sim] loading MuJoCo model {args.xml} ...", flush=True)
    sim = SimNova2(args.xml)
    sim.reset()
    print(f"[sim] home CAD joints: {np.round(sim.home_joints_cad(), 4).tolist()}", flush=True)
    print(f"[sim] home state real: {np.round(sim.state_real(), 2).tolist()}", flush=True)

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logf = open(args.log, "w", encoding="utf-8")
    frame_dir = "/tmp/sim_frames"
    if args.save_frames_every > 0:
        os.makedirs(frame_dir, exist_ok=True)

    def build_obs_frame():
        """dataset 格式 obs（sync 模式用）。"""
        front = sim.render("front")      # (448,448,3) uint8
        wrist = sim.render("wrist")      # (640,480,3) uint8
        state = sim.state_real()
        return {
            "observation.images.front": front,
            "observation.images.wrist": wrist,
            "observation.state": state,
        }

    def log_tick(tick, act, t_start):
        state = sim.state_real()
        rec = {
            "tick": tick,
            "action": np.round(act, 4).tolist(),
            "joints_deg": np.round(state[:6], 3).tolist(),
            "tcp_mm": np.round(state[6:12], 2).tolist(),
            "gripper": round(float(state[12]), 2),
            "block_mm": np.round(sim.block_pos_real_mm(), 2).tolist(),
            "attached": sim.block_attached,
        }
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if tick % 50 == 0:
            print(f"[sim] tick {tick}: action={rec['action']} tcp={rec['tcp_mm']} "
                  f"({time.time()-t_start:.0f}s)", flush=True)
        if args.save_frames_every and tick % args.save_frames_every == 0:
            from PIL import Image
            for cam in ("front", "wrist"):
                Image.fromarray(sim.render(cam)).save(f"{frame_dir}/{tick:05d}_{cam}.png")
        return rec

    # 预热一次（sync 路径，验证策略能出动作）
    with torch.inference_mode():
        obs0 = build_obs_frame()
        obs0 = prepare_observation_for_inference(obs0, device, TASK, "dobot_nova2")
        obs0 = pre(obs0)
        a0 = post(policy.select_action(obs0))
        print(f"[sim] smoke action shape={tuple(a0.shape)} first={np.round(a0[0].cpu().numpy(), 3).tolist()}", flush=True)
        policy.reset()
    sim.reset()

    print(f"[sim] rollout {args.steps} ticks @ {CONTROL_DT}s/step (inference={args.inference}) ...", flush=True)
    t_start = time.time()
    n_infer = 0
    actions_all = []

    if args.inference == "rtc":
        print("[sim] RTC async inference mode", flush=True)
        from lerobot.policies.rtc.action_queue import ActionQueue  # noqa: F401
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.rollout.inference.rtc import RTCInferenceEngine
        from lerobot.utils.feature_utils import hw_to_dataset_features

        class _FakeRobot:
            """RTC 引擎只用 robot_type；SmolVLA 是 delta 动作，不触发相对动作分支。"""
            robot_type = "dobot_nova2"
            action_features = {}

        # 仿真 obs 用 robot 格式（joint_1.pos… / front / wrist），经 hw_features 转成数据集格式
        obs_features_hw = {
            "joint_1.pos": float, "joint_2.pos": float, "joint_3.pos": float,
            "joint_4.pos": float, "joint_5.pos": float, "joint_6.pos": float,
            "tcp.x": float, "tcp.y": float, "tcp.z": float,
            "tcp.rx": float, "tcp.ry": float, "tcp.rz": float,
            "gripper.pos": float,
            "front": (448, 448, 3), "wrist": (640, 480, 3),
        }
        hw_features = hw_to_dataset_features(obs_features_hw, "observation")
        STATE_NAMES = ["joint_1.pos", "joint_2.pos", "joint_3.pos", "joint_4.pos",
                       "joint_5.pos", "joint_6.pos", "tcp.x", "tcp.y", "tcp.z",
                       "tcp.rx", "tcp.ry", "tcp.rz", "gripper.pos"]

        def build_obs_frame_robot():
            st = sim.state_real()
            obs = {n: float(v) for n, v in zip(STATE_NAMES, st)}
            obs["front"] = sim.render("front")
            obs["wrist"] = sim.render("wrist")
            return obs

        policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=10)
        policy.init_rtc_processor()
        engine = RTCInferenceEngine(
            policy, pre, post, _FakeRobot(), policy.config.rtc_config,
            hw_features, TASK, fps=1.0 / CONTROL_DT, device=str(device),
        )
        engine.start()
        engine.resume()   # 置位 _policy_active，启动后台推理
        for tick in range(args.steps):
            t_tick = time.time()
            engine.notify_observation(build_obs_frame_robot())
            action = engine.get_action(None)   # (7,) 或 None
            if action is not None:
                act = action.cpu().numpy().astype(np.float64)
                if act.shape == (7,):
                    n_infer += 1
                    sim.step_delta(act)
                    sim.maybe_attach_block(act)   # 运动学吸附（与训练一致）
                    rec = log_tick(tick, act, t_start)
                    actions_all.append(act)
                    if not np.all(np.isfinite(act)) or not np.all(np.isfinite(rec["tcp_mm"])):
                        print(f"[sim] NaN at tick {tick}, aborting", flush=True)
                        break
            # 按 20Hz 节奏推进，给 RTC 后台线程留出推理时间
            elapsed = time.time() - t_tick
            if elapsed < CONTROL_DT:
                time.sleep(CONTROL_DT - elapsed)
        engine.stop()
    else:
        # sync 模式：select_action 内部管理队列
        for tick in range(args.steps):
            with torch.inference_mode():
                obs = build_obs_frame()
                obs = prepare_observation_for_inference(obs, device, TASK, "dobot_nova2")
                obs = pre(obs)
                action = post(policy.select_action(obs))   # 队列空时内部重新推理
                n_infer += 1
            act = action[0].cpu().numpy()
            sim.step_delta(act)
            sim.maybe_attach_block(act)   # 运动学吸附（与训练一致）
            rec = log_tick(tick, act, t_start)
            actions_all.append(act)
            if not np.all(np.isfinite(act)) or not np.all(np.isfinite(rec["tcp_mm"])):
                print(f"[sim] NaN at tick {tick}, aborting", flush=True)
                break

    logf.close()
    arr = np.array(actions_all)
    print(f"\n[sim] DONE {len(actions_all)} ticks, {n_infer} action steps "
          f"({len(actions_all)/(time.time()-t_start):.2f} Hz wall)")
    if len(arr):
        print(f"[sim] action mean={np.round(arr.mean(0), 3).tolist()} "
              f"std={np.round(arr.std(0), 3).tolist()}")
    print(f"[sim] log: {args.log}")


if __name__ == "__main__":
    main()
