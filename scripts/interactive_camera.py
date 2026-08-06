#!/usr/bin/env python3
"""
实时交互调整 front 相机（自包含，不需 torch）：
显示 front 相机视图，键盘实时调整相机参数，S 保存到 _camera_params.json。

控件:
  ←/→        相机 x 位置 (0.02m/次)
  ↑/↓        相机 z 高度
  W/S        相机 y 位置(前后)
  Q/E        俯角 pitch (与垂直夹角, 度)
  A/D        偏航 yaw (绕Z)
  Z/X        roll (绕视轴)
  R/F        fovy
  S          保存相机参数
  ESC        退出
"""
import json, os, sys
import numpy as np
os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco
import cv2

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nova2_scene_corrected.xml")

# ---- 坐标变换 / IK（与 sim_smolvla_rollout 一致） ----
R_BASE = np.diag([-1.0, -1.0, 1.0])
T_BASE = np.array([3.7, 17.1, 133.9])
GRIPPER_OPEN_M = 0.0721
IK_LAMBDA = 0.01
IK_ITERS = 3


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


# ---- 加载模型并设置抓取姿态 ----
m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
tcp_id = m.site("tcp").id

# 方块：最新真实抓取位置, 30mm 高放桌面
block_real = np.array([-138.1, -469.7, 79.0])
block_cad = R_BASE @ (block_real * 1e-3 - T_BASE * 1e-3)
block_cad[1] += 0.10  # 工作区前移10cm(机械臂远离工作台)
block_cad[2] = -0.079
bq = m.jnt_qposadr[m.body("block").jntadr[0]]
d.qpos[bq:bq + 3] = block_cad
d.qpos[bq + 3:bq + 7] = np.array([1.0, 0, 0, 0])


def tcp_cad():
    pos = d.site_xpos[tcp_id].copy()
    mat = d.site_xmat[tcp_id].reshape(3, 3).copy()
    return pos, mat


def jacobian():
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, jacr, tcp_id)
    return np.vstack([jacp[:, :6], jacr[:, :6]])


def ik_to_q(target_pos_cad, target_mat_cad):
    q = d.qpos[:6].copy()
    lam2 = IK_LAMBDA ** 2
    for _ in range(IK_ITERS):
        d.qpos[:6] = q
        mujoco.mj_forward(m, d)
        pos, mat = tcp_cad()
        e_pos = target_pos_cad - pos
        e_rot = rot_to_axis_angle(target_mat_cad @ mat.T)
        e = np.concatenate([e_pos, e_rot])
        J = jacobian()
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
        q = q + dq
    return q


tcp_real = np.array([-138.1, -469.7, 31.4])
rpy = np.array([-175.37, -2.5, 171.11])
tcp_cad_target = R_BASE @ (tcp_real * 1e-3 - T_BASE * 1e-3)
tcp_cad_target[1] += 0.10  # 工作区前移10cm
tcp_mat_target = R_BASE @ zyx_euler_to_rot(rpy)
q = ik_to_q(tcp_cad_target, tcp_mat_target)
d.qpos[:6] = q
fq = m.joint("finger_left_joint").qposadr[0]
d.qpos[fq] = GRIPPER_OPEN_M
d.qpos[fq + 1] = GRIPPER_OPEN_M
mujoco.mj_forward(m, d)

# ---- 相机参数（初值 = 补偿后，使渲染位置 = 用户期望视角） ----
pos = np.array([-0.5974, 0.7634, 0.5312])
yaw = 119.0
pitch = 34.6
roll = 84.0
fovy = 46.0
CAM_OFFSET = 0.034  # MuJoCo 固定相机烘焙偏移系数(沿right向量), 交互渲染需补偿


# 渲染器 + 场景相机(可运行时直改, 不重编译)
_renderer = mujoco.Renderer(m, 448, 448)
_renderer.update_scene(d, camera="front")
_sc = _renderer.scene.camera[0]
_near = float(_sc.frustum_near)


def render():
    a = np.radians(pitch)
    y = np.radians(yaw)
    look = np.array([np.sin(a) * np.sin(y), np.sin(a) * np.cos(y), -np.cos(a)])
    cam_z = -look
    cam_x = np.cross(np.array([0.0, 0.0, 1.0]), cam_z)
    n = np.linalg.norm(cam_x)
    cam_x = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else cam_x / n
    cam_y = np.cross(cam_z, cam_x)
    r = np.radians(roll)
    c, s = np.cos(r), np.sin(r)
    cam_xr = c * cam_x + s * cam_y
    cam_yr = -s * cam_x + c * cam_y
    # 补偿 MuJoCo 固定相机烘焙偏移: sc.pos = cam_pos - CAM_OFFSET*cam_xr
    _sc.pos = np.asarray(pos, dtype=float) - CAM_OFFSET * cam_xr
    _sc.forward = look
    _sc.up = cam_yr
    half = _near * np.tan(np.radians(fovy / 2.0))
    _sc.frustum_top = half
    _sc.frustum_bottom = -half
    _sc.frustum_width = 2.0 * half
    return _renderer.render().copy()


HELP = (
    "W/S y前后  A/D x左右  Q/E z高度\n"
    "Z/X roll  C/V yaw  B/N pitch\n"
    "[/] fovy  P保存  ESC退出"
)

print("交互调整 front 相机（先点击窗口获得焦点）:")
print(HELP)

if "--test" in sys.argv:
    img = render()
    print("[test] 初始 mean %.1f block_px %d" % (
        img.mean(), int(((img[..., 0] > 150) & (img[..., 1] > 90) & (img[..., 2] < 120)).sum())))
    pos[0] += 0.1
    img1 = render()
    d1 = abs(img1 - img).mean()
    roll += 5.0
    img2 = render()
    d2 = abs(img2 - img1).mean()
    fovy -= 5.0
    img3 = render()
    d3 = abs(img3 - img2).mean()
    print("[test] x+0.1 diff %.1f  roll+5 diff %.1f  fovy-5 diff %.1f" % (d1, d2, d3))
    sys.exit(0)

while True:
    img = render()
    info = (f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
            f"pitch={pitch:.1f} yaw={yaw:.1f} roll={roll:.1f} fovy={fovy:.0f}")
    img2 = img.copy()
    for i, line in enumerate(HELP.split("\n")):
        cv2.putText(img2, line, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(img2, info, (10, 448 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imshow("front camera", img2)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break
    elif key == ord('p'):
        data = {"pos": pos.tolist(), "yaw": yaw, "pitch": pitch, "roll": roll, "fovy": fovy}
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_camera_params.json"), "w") as f:
            json.dump(data, f, indent=2)
        print("[保存] _camera_params.json", data)
    elif key == ord('w'):
        pos[1] += 0.02
    elif key == ord('s'):
        pos[1] -= 0.02
    elif key == ord('a'):
        pos[0] -= 0.02
    elif key == ord('d'):
        pos[0] += 0.02
    elif key == ord('q'):
        pos[2] += 0.02
    elif key == ord('e'):
        pos[2] -= 0.02
    elif key == ord('z'):
        roll += 2.0
    elif key == ord('x'):
        roll -= 2.0
    elif key == ord('c'):
        yaw += 2.0
    elif key == ord('v'):
        yaw -= 2.0
    elif key == ord('b'):
        pitch += 2.0
    elif key == ord('n'):
        pitch -= 2.0
    elif key == 91:  # [
        fovy += 2.0
    elif key == 93:  # ]
        fovy -= 2.0

cv2.destroyAllWindows()
print("退出。")
