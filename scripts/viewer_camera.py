#!/usr/bin/env python3
"""
进入 MuJoCo 3D 仿真界面手动调整 front 相机视角。
- 打开 3D 视图：鼠标左键旋转、右键平移、滚轮缩放
- 找到想要的视角后按 S 保存当前视角为 front 相机参数 -> _viewer_cam.json
- 关闭窗口退出（或按 ESC）
"""
import json, os, sys
import numpy as np
os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco
import mujoco.viewer

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nova2_scene_corrected.xml")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_viewer_cam.json")

# ---- 坐标变换 / IK ----
R_BASE = np.diag([-1.0, -1.0, 1.0])
T_BASE = np.array([3.7, 17.1, 133.9])


def zyx_euler_to_rot(rpy_deg):
    r, p, y = np.radians(rpy_deg)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_to_axis_angle(R):
    R = np.asarray(R, dtype=float)
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * np.sin(angle)) * angle


# ---- 加载模型 + 抓取姿态 ----
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
tcp_id = m.site("tcp").id
bq = m.jnt_qposadr[m.body("block").jntadr[0]]
bc = R_BASE @ (np.array([-138.1, -469.7, 79.0]) * 1e-3 - T_BASE * 1e-3)
bc[1] += 0.10  # 工作区前移10cm(机械臂远离工作台)
bc[2] = -0.079
d.qpos[bq:bq + 3] = bc
d.qpos[bq + 3:bq + 7] = np.array([1.0, 0, 0, 0])


def ik_to_q(target_pos_cad, target_mat_cad):
    q = d.qpos[:6].copy()
    for _ in range(3):
        d.qpos[:6] = q
        mujoco.mj_forward(m, d)
        pos = d.site_xpos[tcp_id].copy()
        mat = d.site_xmat[tcp_id].reshape(3, 3).copy()
        e_pos = target_pos_cad - pos
        e_rot = rot_to_axis_angle(target_mat_cad @ mat.T)
        e = np.concatenate([e_pos, e_rot])
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, tcp_id)
        J = np.vstack([jp[:, :6], jr[:, :6]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q = q + dq
    return q


tcp_target = R_BASE @ (np.array([-138.1, -469.7, 31.4]) * 1e-3 - T_BASE * 1e-3)
tcp_target[1] += 0.10  # 工作区前移10cm
tmat = R_BASE @ zyx_euler_to_rot(np.array([-175.37, -2.5, 171.11]))
q = ik_to_q(tcp_target, tmat)
d.qpos[:6] = q
fq = m.joint("finger_left_joint").qposadr[0]
d.qpos[fq] = 0.0721
d.qpos[fq + 1] = 0.0721
mujoco.mj_forward(m, d)

if "--test" in sys.argv:
    tcp = d.site_xpos[tcp_id].copy()
    blk = d.body("block").xpos.copy()
    print("[test] tcp", np.round(tcp, 3), " block", np.round(blk, 3))
    print("[test] 模型加载+IK 成功")
    sys.exit(0)

print("MuJoCo 3D 仿真界面:")
print("  鼠标左键=旋转视角  右键=平移  滚轮=缩放")
print("  找到满意视角后按 S 保存为 front 相机，然后关窗")
print("  当前视角参数会实时显示在窗口左下角")


def save_camera():
    az = np.radians(viewer.cam.azimuth)
    el = np.radians(viewer.cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = np.array(viewer.cam.lookat) - viewer.cam.distance * fwd
    cam_z = -fwd
    cam_x = np.cross(np.array([0.0, 0.0, 1.0]), cam_z)
    n = np.linalg.norm(cam_x)
    cam_x = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else cam_x / n
    cam_y = np.cross(cam_z, cam_x)
    xy = " ".join(f"{v:.4f}" for v in list(cam_x) + list(cam_y))
    data = {
        "pos": [round(float(v), 4) for v in pos],
        "xyaxes": xy,
        "fovy": 45.0,
        "lookat": list(viewer.cam.lookat),
        "distance": float(viewer.cam.distance),
        "azimuth": float(viewer.cam.azimuth),
        "elevation": float(viewer.cam.elevation),
    }
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    print("[保存] front 相机参数 ->", OUT)
    print("  pos", data["pos"], " fovy", data["fovy"])
    print("  xyaxes", xy)


def on_key(key):
    if key == 83:  # GLFW_KEY_S
        save_camera()


viewer = mujoco.viewer.launch_passive(m, d, key_callback=on_key)
# 初始视角 = front 相机当前定义（这样重开能看到之前保存的视角）
try:
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    front_pos = np.array(m.cam_pos[fid]).copy()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.cam_quat[fid])
    R = R.reshape(3, 3)
    look = -R[:, 2]
    dist = 1.2
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = front_pos + look * dist
    viewer.cam.distance = dist
    viewer.cam.elevation = float(np.degrees(np.arcsin(np.clip(look[2], -1, 1))))
    viewer.cam.azimuth = float(np.degrees(np.arctan2(look[1], look[0])))
    while viewer.is_running():
        mujoco.mj_forward(m, d)
        viewer.sync()
finally:
    viewer.close()
print("退出。")
