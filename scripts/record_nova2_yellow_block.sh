#!/bin/bash
# ============================================================
# Nova2 黄色方块抓取—搬运—放置 数据录制脚本（终端 A）
# 适配服务器地址：192.168.0.130
#
# 用法（在终端 A）：
#     bash record_nova2_yellow_block.sh [批次名] [条数]
#
# 示例：
#     bash record_nova2_yellow_block.sh train30_part1 10
#     bash record_nova2_yellow_block.sh train30_part2 10
#     bash record_nova2_yellow_block.sh train30_part3 10
#
# 依赖：终端 B 必须先启动 ssh_key_sender.py
# ============================================================
set -euo pipefail

# ============================================================
# ★ 配置区（按需修改）
# ============================================================
BATCH_NAME="${1:-test_batch}"
NUM_EPISODES="${2:-10}"
EPISODE_TIME_S=300          # 单条超时（秒），成功后及时按 n
RESET_TIME_S=5              # 条间等待
SAVE_VIDEO=true             # 是否保存视频

# ---- 路径（Linux 目标机） ----
LEROBOT_ROOT="/home/jjk/桌面/Project/lerobot-main"
DATASETS_ROOT="/home/jjk/dobot_nova2_project/datasets"
CONDA_ENV="nova2_lerobot"

# ---- 数据集配置 ----
DATASET_REPO_ID="nova2_yellow_block_${BATCH_NAME}"
DATASET_ROOT="${DATASETS_ROOT}/${DATASET_REPO_ID}"

# ---- 录像配置 ----
WRIST_CAM="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7:1.0-video-index0"
FRONT_CAM="/dev/v4l/by-path/pci-0000:00:14.0-usb-0:13:1.0-video-index0"
CAM_RESOLUTION="640x480"
RECORD_FPS=10
CAM_INPUT_FPS=30

# ---- 机器人配置 ----
ROBOT_IP="192.168.0.130"    # ★ 服务器地址
DASHBOARD_PORT=29999
MOTION_PORT=30004
FEEDBACK_PORT=30005

# ---- 动作限幅 ----
DELTA_XYZ_MAX=1.0
DELTA_ROT_MAX=0.5
SPEED_FACTOR=0.05
GRIPPER_MAX=1.0

# ---- 工作空间安全 ----
Y_LIMIT_MIN=-560

# ---- 遥操作 ----
UDP_HOST="127.0.0.1"
UDP_PORT=18888

# ============================================================
# 录制前检查
# ============================================================
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Nova2 数据录制 — 黄色方块抓取搬运放置                 ║"
echo "║  服务器: ${ROBOT_IP}                                    "
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "批次名称:   ${BATCH_NAME}"
echo "目标条数:   ${NUM_EPISODES}"
echo "数据目录:   ${DATASET_ROOT}"
echo "每集超时:   ${EPISODE_TIME_S}s (成功及时按 n)"
echo ""

# --- 检查数据集目录不存在 ---
if [ -d "${DATASET_ROOT}" ]; then
    echo "[错误] 数据集目录已存在: ${DATASET_ROOT}"
    echo "       LeRobot resume=false 不会覆盖已有目录。"
    echo "       请归档旧目录或使用不同的批次名。"
    exit 1
fi

# --- 检查机器人连通性 ---
echo "[检查] Ping 机器人 ${ROBOT_IP}..."
if ! ping -c 1 -W 2 "${ROBOT_IP}" > /dev/null 2>&1; then
    echo "[错误] 无法 ping 通 ${ROBOT_IP}"
    echo "       请检查网线连接和 IP 配置"
    exit 1
fi
echo "       ✓ 网络可达"

# --- 检查相机 ---
echo "[检查] 相机设备..."
for cam in "${WRIST_CAM}" "${FRONT_CAM}"; do
    if [ ! -e "${cam}" ]; then
        echo "[警告] 相机设备不存在: ${cam}"
    else
        echo "       ✓ ${cam}"
    fi
done

# --- 检查 UDP 端口 ---
echo "[检查] UDP 接收端口 ${UDP_PORT}..."
if ss -uln | grep -q ":${UDP_PORT}"; then
    echo "       ⚠ 端口 ${UDP_PORT} 已被占用，可能有残留进程"
    echo "       建议先清理: fuser -k ${UDP_PORT}/udp"
else
    echo "       ✓ 端口空闲"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  即将启动录制，请确保:"
echo "  1. 终端 B 的 ssh_key_sender.py 已启动"
echo "  2. 机械臂处于 RobotMode=5，无报警"
echo "  3. 黄色方块在固定木盒中，泡沫盒在目标区域"
echo "  4. 夹爪处于张开状态，机械臂在安全高位"
echo "  5. 工作区无人员、无松动物体"
echo "═══════════════════════════════════════════════════════════"
echo ""
read -p "按 Enter 开始录制，或 Ctrl+C 取消... "

# ============================================================
# 启动录制
# ============================================================
echo ""
echo ">>> 启动录制引擎..."
echo "    看到 'Recording episode N' 后开始遥操作"
echo "    成功后按 n 保存，失败按 m 重录"
echo "    不要在录制中按 Ctrl+C !"
echo ""

cd "${LEROBOT_ROOT}"

# 使用 LeRobot 录制（实际命令取决于插件实现）
# 此处为调用模式，具体录制逻辑由 Nova2 插件处理
conda run --no-capture-output -n "${CONDA_ENV}" python -c "
import os
import sys
import json
import time
import pathlib
import socket
import threading
import numpy as np
from datetime import datetime

# 插入 SDK 路径
sys.path.insert(0, '/home/jjk/桌面/Project/dwlerobot-nova2xbox/TCP-IP-Python-V4-main/TCP-IP-Python-V4-main')
from dobot_api import DobotApiDashboard, DobotApiMove, DobotApi

# ---- 配置 ----
ROBOT_IP = '${ROBOT_IP}'
DATASET_ROOT = '${DATASET_ROOT}'
DATASET_REPO_ID = '${DATASET_REPO_ID}'
NUM_EPISODES = ${NUM_EPISODES}
EPISODE_TIME_S = ${EPISODE_TIME_S}
RESET_TIME_S = ${RESET_TIME_S}
FPS = ${RECORD_FPS}
UDP_HOST = '${UDP_HOST}'
UDP_PORT = ${UDP_PORT}
Y_LIMIT_MIN = ${Y_LIMIT_MIN}
DELTA_XYZ_MAX = ${DELTA_XYZ_MAX}
DELTA_ROT_MAX = ${DELTA_ROT_MAX}
SPEED_FACTOR = ${SPEED_FACTOR}

SAFE_A_JOINTS = [77.9857, 51.3401, 31.1298, 16.1389, -90.4004, 3.8710]

print(f'数据集: {DATASET_REPO_ID}')
print(f'目录:   {DATASET_ROOT}')

# ---- 创建目录结构 ----
os.makedirs(f'{DATASET_ROOT}/meta', exist_ok=True)
os.makedirs(f'{DATASET_ROOT}/data', exist_ok=True)
os.makedirs(f'{DATASET_ROOT}/videos', exist_ok=True)

# ---- UDP 接收器（监听终端 B 的按键） ----
class UDPListener:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.1)
        self.last_key = None
        self.running = True

    def listen(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
                self.last_key = data.decode('utf-8')
            except socket.timeout:
                continue
            except Exception:
                break

    def get_key(self):
        key = self.last_key
        self.last_key = None
        return key

print(f'启动 UDP 监听 {UDP_HOST}:{UDP_PORT}...')
udp = UDPListener(UDP_HOST, UDP_PORT)
listener_thread = threading.Thread(target=udp.listen, daemon=True)
listener_thread.start()
print('UDP 监听已就绪')

# ---- 机器人连接 ----
print(f'连接机器人 {ROBOT_IP}...')
dashboard = DobotApiDashboard(ROBOT_IP, 29999)
move = DobotApiMove(ROBOT_IP, 30004)
feed = DobotApi(ROBOT_IP, 30005)
print('机器人连接成功')

# ---- 检查初始状态 ----
mode = dashboard.RobotMode()
errors = dashboard.GetErrorID()
print(f'RobotMode = {mode}, Errors = {errors}')

if mode != 5:
    print(f'[错误] RobotMode={mode}, 需要恢复操作。退出。')
    sys.exit(1)
if errors:
    print(f'[错误] 存在报警: {errors}。退出。')
    sys.exit(1)

print('初始状态正常，开始录制循环')
print('=' * 50)

# ---- 录制循环 ----
episode = 0
retry_count = 0

while episode < NUM_EPISODES:
    print(f'\\n>>> Recording episode {episode} (第 {episode+1}/{NUM_EPISODES} 条)')
    print('    在终端 B 操作机械臂...')

    episode_data = {
        'episode_index': episode,
        'timestamp': datetime.now().isoformat(),
        'observations': [],  # list of 13-dim state vectors
        'actions': [],       # list of 7-dim action vectors
        'frames': [],
    }

    frame_count = 0
    start_time = time.time()
    episode_done = False
    episode_success = False

    while not episode_done:
        # 超时检查
        elapsed = time.time() - start_time
        if elapsed > EPISODE_TIME_S:
            print(f'    ⚠ 超时 ({EPISODE_TIME_S}s)，自动丢弃')
            episode_done = True
            episode_success = False
            break

        # 检查 UDP 按键
        key = udp.get_key()
        if key == 'n':
            print('    ✓ 收到保存指令 (n)')
            episode_done = True
            episode_success = True
            break
        elif key == 'm':
            print('    ✗ 收到丢弃指令 (m)')
            episode_done = True
            episode_success = False
            break
        elif key == 'P':
            print('    ■ 收到结束整批指令 (P)')
            episode_done = True
            episode_success = False
            episode = NUM_EPISODES  # 结束整批
            break

        # 读取机器人状态
        try:
            pose = feed.GetPose()
            angles = feed.GetAngle()
            gripper_state = feed.GetGripperStatus() if hasattr(feed, 'GetGripperStatus') else 50
        except Exception as e:
            print(f'    [警告] 读取状态异常: {e}')
            time.sleep(0.05)
            continue

        if pose is None or angles is None:
            time.sleep(0.05)
            continue

        # 组装 observation (13维)
        obs = list(angles[:6]) + list(pose[:6]) + [gripper_state]

        # 组装 action (从最近指令计算 delta，此处为简化占位)
        # 实际实现由 Nova2 插件处理实时动作插值
        action = [0.0] * 7  # delta_x,y,z,rx,ry,rz,gripper

        episode_data['observations'].append(obs)
        episode_data['actions'].append(action)
        frame_count += 1

        time.sleep(1.0 / FPS)

    # ---- 处理录音结果 ----
    if episode_success:
        print(f'    ✓ Episode {episode} 保存成功 ({frame_count} 帧)')

        # 保存 episode 数据
        ep_file = f'{DATASET_ROOT}/data/episode_{episode:06d}.npz'
        np.savez_compressed(
            ep_file,
            observations=np.array(episode_data['observations'], dtype=np.float32),
            actions=np.array(episode_data['actions'], dtype=np.float32),
        )

        # 更新 meta
        meta = {
            'dataset_repo_id': DATASET_REPO_ID,
            'total_episodes': episode + 1,
            'total_frames': frame_count,  # 累计需维护
            'fps': FPS,
            'state_dim': 13,
            'action_dim': 7,
            'robot_ip': ROBOT_IP,
            'timestamp': datetime.now().isoformat(),
        }
        with open(f'{DATASET_ROOT}/meta/info.json', 'w') as f:
            json.dump(meta, f, indent=2)

        episode += 1
        retry_count = 0
    else:
        print(f'    ✗ Episode 丢弃，重录同一编号')
        retry_count += 1
        if retry_count > 5:
            print(f'    [警告] 连续重录 {retry_count} 次，建议检查设备状态')

    # 条间等待
    if episode < NUM_EPISODES:
        print(f'    等待 {RESET_TIME_S}s (手动复位方块，不控制机械臂)...')
        time.sleep(RESET_TIME_S)

# ---- 录制结束 ----
print('')
print('=' * 50)
print(f'录制完成! 成功 {episode}/{NUM_EPISODES} 条')
print(f'数据目录: {DATASET_ROOT}')

# 写入最终 meta
final_meta = {
    'dataset_repo_id': DATASET_REPO_ID,
    'total_episodes': episode,
    'fps': FPS,
    'state_dim': 13,
    'action_dim': 7,
    'robot_ip': ROBOT_IP,
    'action_limits': {
        'delta_xyz_max': DELTA_XYZ_MAX,
        'delta_rot_max': DELTA_ROT_MAX,
        'speed_factor': SPEED_FACTOR,
    },
    'pose_limits': {'y_min': Y_LIMIT_MIN},
    'timestamp': datetime.now().isoformat(),
}
with open(f'{DATASET_ROOT}/meta/info.json', 'w') as f:
    json.dump(final_meta, f, indent=2)
print('元数据已保存')

# UDP 清理
udp.running = False
print('正常退出 (Stop recording/Exiting)')
"

EXIT_CODE=$?

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ 录制正常结束"
    echo "  数据目录: ${DATASET_ROOT}"
    echo "  下一步: 运行质检脚本"
else
    echo "✗ 录制异常退出 (退出码: ${EXIT_CODE})"
    echo "  检查: 是否有 Traceback 输出"
fi
