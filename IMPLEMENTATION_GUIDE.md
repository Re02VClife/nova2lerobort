# Nova2 黄色方块抓取—搬运—放置 实现指南

> **最后更新：2026-07-14 — 已通过实机验证**

---

## 一、网络拓扑（已确认）

```
┌──────────────────────────────────────────────────────┐
│  路由器 192.168.0.1                                    │
│    ├── 192.168.0.100  — 机器人 V4 控制器 (LAN口)       │
│    ├── 192.168.0.112  — Windows 电脑                   │
│    └── 192.168.0.130  — Linux 主机 WiFi (wlxc8...)     │
└──────────────────────────────────────────────────────┘
              │
              │ SSH (ssh jjk@192.168.0.130)
              ▼
┌──────────────────────────────────────────────────────┐
│  Linux 主机 (jjk@ym, RTX 4090D)                       │
│                                                       │
│  无线: wlxc83a35368848 → 192.168.0.130/24 (连路由器)   │
│        ↑ Windows 通过这个 IP SSH 进来                  │
│                                                       │
│  有线: enp4s0 → 192.168.5.10/24 (直连机器人控制器)      │
│        ↑ 机械臂 SDK 通过这个网卡通信                    │
│                                                       │
│  机器人控制器: 192.168.5.1 ← ★ 与交接手册一致！        │
│        enp4s0 延迟仅 0.47ms                            │
└──────────────────────────────────────────────────────┘
              │ TCP/IP (SDK 端口 29999/30004/30005)
              ▼
┌──────────────────────────────────────────────────────┐
│  Dobot Nova2 V4 控制器                                 │
│  相机: 腕部 + 前视, 640×480, 录制 10fps               │
│  夹爪: 平行夹爪, Modbus Tool 485                       │
└──────────────────────────────────────────────────────┘
```

---

## 二、数据接口定义

| 项目 | 维度 | 内容 |
|------|------|------|
| **Observation (State)** | 13 | 6关节角 + 6 TCP位姿 + 1夹爪状态 |
| **Action** | 7 | delta_x/y/z(mm) + delta_rx/ry/rz(°) + gripper(-1/0/+1) |
| **动作限幅** | — | XYZ ≤ 1.0mm/帧, 旋转 ≤ 0.5°/帧, 速度因子 5% |

---

## 三、操作全流程

### 3.1 从 Windows SSH 连接 Linux 主机

在 Windows 上打开 **两个** PowerShell/Terminal 窗口：

```powershell
# 两个窗口都执行：
ssh jjk@192.168.0.130
```

- **窗口 1 → 终端 B**（键盘发送器）
- **窗口 2 → 终端 A**（录制监控）

> 首次连接会提示 ED25519 指纹，输入 `yes` 然后输入密码。

---

### 3.2 终端 B — 检查机器人状态（先不动，只读）

```bash
conda activate nova2_lerobot
python -c "
import sys
sys.path.insert(0, '/home/jjk/桌面/Project/dwlerobot-nova2xbox/TCP-IP-Python-V4-main/TCP-IP-Python-V4-main')
from dobot_api import DobotApiDashboard

d = DobotApiDashboard('192.168.5.1', 29999)
print('Mode   =', d.RobotMode())
print('Error  =', d.GetErrorID())
print('Angles =', d.GetAngle())
pose = d.GetPose()
print('Pose   =', pose)
print('TCP_Y  = {:.1f} mm (下限 -560)'.format(pose[1]))
"
```

| 检查项 | 合格标准 | 当前实测 |
|--------|---------|----------|
| RobotMode | = 5 | ✅ 5 |
| ErrorID | 空（无报警） | ✅ [] |
| TCP_Y | ≥ -560 mm | ✅ -398.3 |

---

### 3.3 终端 B — 启动键盘发送器（先启动）

```bash
conda activate nova2_lerobot
python ~/dobot_nova2_project/scripts/ssh_key_sender.py
```

看到 "发送器已就绪"、按键表打印后，**焦点留在此窗口**。

---

### 3.4 终端 A — 启动录制脚本（后启动）

```bash
conda activate nova2_lerobot

# 查看已有哪些批次
ls -d ~/dobot_nova2_project/datasets/nova2_yellow_block_train30_part*
cat ~/dobot_nova2_project/datasets/VALID_EPISODES_CURRENT.txt 2>/dev/null | head -20

# 启动录制（示例：part3）
bash ~/dobot_nova2_project/scripts/record_nova2_yellow_block_train30_part3.sh
```

看到 `"Recording episode 0"` 后，开始在**终端 B** 按键操作。

---

### 3.5 安全确认：按键不会冲出去

每个按键只发送 **一条** UDP 消息到录制进程，移动量已硬编码限制：

| 参数 | 值 | 含义 |
|------|----|------|
| XYZ_STEP | 1.0 mm | 每按 1 次走 1 毫米 |
| ROT_STEP | 0.5° | 每按 1 次转 0.5 度 |
| SPEED_FACTOR | 5 | 全速的 5% |
| 精调模式 (按1) | ×0.33 | **0.33 mm/步** |
| 正常模式 (按2) | ×1.0 | 1.0 mm/步 |
| 粗调模式 (按3) | ×1.0 | 和正常一样 |
| MAX_XYZ_STEP | 1.0 mm | 硬限幅，超不过 |
| Y 硬限 | -560 mm | 达到即停 |

> **结论：按一下走一步，长按不会连续飞。按空格立即停止。**

---

### 3.6 按键速查

| 按键 | 功能 | 注意 |
|------|------|------|
| W/S | Y-/Y+ | **W = 远离底座，危险方向！** |
| A/D | X+/X- | 水平横移 |
| R/F | Z+/Z- | R 抬高, F 下降 |
| I/K | Rx+/Rx- | 绕 X 旋转 |
| J/L | Ry+/Ry- | 绕 Y 旋转 |
| Q/E | Rz+/Rz- | 绕 Z 旋转 |
| 1 / 2 / 3 | 精调 / 正常 / 粗调 | 精调 = 0.33mm/步 |
| 空格 | **立即停止归零** | 紧急情况下第一反应 |
| [ / ] | 闭合 / 张开夹爪 | |
| **n** | ✅ 保存当前 episode | 成功轨迹 |
| **m** | ❌ 丢弃并重录 | 失败轨迹 |
| P | 正常结束整批 | 确需提前终止时使用 |

---

### 3.7 标准抓取轨迹

```
1. 安全高位（当前位姿即可）
2. 下降 → 接近黄色方块
3. [ 闭合夹爪，抓稳
4. R 抬高 → 避免低位横移
5. A/D 水平搬运到泡沫盒上方
6. F 下降 → ] 张开夹爪释放
7. R 抬高回安全位
8. 终端 B 按 n 保存
```

---

### 3.8 操作节奏

- 先在高位调 X/Y，再下降
- 接近方块和泡沫盒时**按 1 切换精调模式**（0.33mm/步）
- 接近 Y=-560 边界时**短按**，频繁按空格归零
- 完成任务后**及时按 n**，减少无意义静止尾帧
- 失败（掉落、碰撞、抓空、未释放）→ 按 **m** 丢弃重录
- **不要按 Ctrl+C 中断录制！**

---

## 四、数据质检

```bash
# 每批录完后
DATA="~/dobot_nova2_project/datasets/nova2_yellow_block_train30_part1"

# 元数据
cat "$DATA/meta/info.json"

# 检查目录大小
du -sh "$DATA"

# 视频列表
find "$DATA/videos" -type f | sort
```

| 检查项 | 合格标准 | 不合格处理 |
|--------|---------|-----------|
| gripper_actions | 包含 -1(闭合)、0(保持)、+1(张开) | 判为失败 |
| 动作 XYZ | ≤ 1.0 mm/帧 | 判为异常 |
| 动作旋转 | ≤ 0.5°/帧 | 判为异常 |
| 数值 | 无 NaN/Inf | **禁止训练** |
| TCP Y | ≥ -560 mm | 安全异常 |
| 视频 | 双相机可解码、任务成功 | 失败重录 |

---

## 五、安全红线

| 规则 | 说明 |
|------|------|
| **不关闭奇异保护** | 奇异是运动学条件，不是可删除的软件区域 |
| **Y ≥ -560 mm** | 工作空间硬限，不得绕过 |
| **报警后先停采** | Error 17（逆解无解）、Error 27（肘奇异） |
| **SAFE_A** | joints=[77.99, 51.34, 31.13, 16.14, -90.40, 3.87] |
| **Mode≠5 不录** | 必须恢复到就绪模式 |

### 故障恢复流程

```
Mode=9 (报警)
  → 读取 GetErrorID() 记录详情
  → 确认现场安全
  → ClearError()
  → 若 Mode=10 (暂停): Stop() → Mode=5
  → 低速 MovJ 使用关节插补回 SAFE_A
  → 确认 Mode=5、无报警后继续
  → 不要连续重发命令！
```

---

## 六、关键路径速查

| 用途 | 路径 |
|------|------|
| LeRobot 主仓库 | `~/桌面/Project/lerobot-main` |
| Nova2 插件（正式，可改） | `~/桌面/Project/lerobot-main/lerobot_robot_dobot_nova2/` |
| Nova2 插件（重复，**禁止改**） | `~/桌面/Project/dwlerobot-nova2xbox/lerobot_robot_dobot_nova2/` |
| Dobot SDK | `~/桌面/Project/dwlerobot-nova2xbox/TCP-IP-Python-V4-main/TCP-IP-Python-V4-main/` |
| 脚本目录 | `~/dobot_nova2_project/scripts/` |
| 数据集根目录 | `~/dobot_nova2_project/datasets/` |
| StarVLA 仓库 | `~/starVLA/` |
| Conda 环境 | `nova2_lerobot` |

---

## 七、目录纪律

```
✅ 只改: ~/桌面/Project/lerobot-main/lerobot_robot_dobot_nova2/
❌ 不改: ~/桌面/Project/dwlerobot-nova2xbox/lerobot_robot_dobot_nova2/
        （重复插件，仅作参考；误改会导致"代码改了但运行未生效"）
```

---

## 八、已有数据集状态

| 数据集 | 状态 |
|--------|------|
| `nova2_yellow_block_formal_v1` | 4ep，ep0/1/2合格，ep3无闭合 |
| `nova2_yellow_block_formal_v1_part2` | 6ep，ep1/2/3/4合格，ep0/5无闭合 |
| `nova2_yellow_block_formal_v1_part3` | 待确认 |
| `nova2_yellow_block_train30_part1` | 已录制 |
| `nova2_yellow_block_train30_part2` | 已录制 |
| `nova2_yellow_block_train30_part3` | **待录制（计划 10 条）** |
| `nova2_yellow_block_train30_merged` | 已合并 |

---

## 九、下一步计划

```
1. ✋ 先遥控练习（不录制），熟悉按键手感
2. ✅ 确认 part1/part2 各自有 10 条合格轨迹
3. ⬜ 录制 part3（10 条）
4. ⬜ 质检 part1-3，筛选有效 episode
5. ⬜ 合并为 merged 数据集
6. ⬜ 计算 Nova2 自己的 state/action 统计量（禁止用 LIBERO 的！）
7. ⬜ StarVLA 首轮训练（过拟合 验证 + 链路确认）
```
