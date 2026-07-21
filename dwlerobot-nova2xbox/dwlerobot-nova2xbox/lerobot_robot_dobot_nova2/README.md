# LeRobot DOBOT Nova2 遥操项目

这个项目是放在本地 LeRobot 仓库里的第三方插件包：

```text
E:\jushenzhineng\lerobot\lerobot_robot_dobot_nova2
```

实现目标：

- 使用 DOBOT 官方 `TCP-IP-Python-V3` SDK 控制 Nova2，不引用之前的旧工程代码。
- 提供 LeRobot `Robot` 插件：`--robot.type=dobot_nova2`。
- 提供匹配动作字段的键盘遥操插件：`--teleop.type=dobot_nova2_keyboard`。
- 为后续在 Nova2 上采集数据、训练/复现 SmolVLA 保留标准 LeRobot action/observation 接口。

## 安装

在 LeRobot 环境里执行：

```powershell
cd E:\jushenzhineng\lerobot
conda activate lerobot
python -m pip install -e .
python -m pip install -e .\lerobot_robot_dobot_nova2
```

也可以只安装 Nova2 插件：

```powershell
cd E:\jushenzhineng\lerobot\lerobot_robot_dobot_nova2
pip install -e .
```

## 官方 SDK 路径

默认使用：

```text
E:\jushenzhineng\DOBOT\TCP-IP-Python-V3-main\TCP-IP-Python-V3-main
```

如果 SDK 移动了，可以通过 CLI 覆盖：

```powershell
--robot.sdk_path="E:\path\to\TCP-IP-Python-V3-main\TCP-IP-Python-V3-main"
```

## 快速遥操

先用项目自带的控制台遥操脚本做低速验证：

```powershell
cd E:\jushenzhineng\lerobot
conda run -n lerobot python -m lerobot_robot_dobot_nova2.teleop_console --host 192.168.5.1 --startup --control-mode movl --speed-factor 5 --xyz-step 1 --rot-step 0.5
```

按键：

```text
W/S: X +/-
A/D: Y +/-
R/F: Z +/-
I/K: Rx +/-
J/L: Ry +/-
U/O: Rz +/-
[/]: 夹爪 关/开
Q: 退出
```

第一次测试建议机械臂周围无人、急停可触达，并保持：

```text
speed_factor=5
xyz_step=1 mm
rot_step=0.5 deg
control_mode=movl
```

## 使用 LeRobot CLI 遥操/记录

安装插件后，LeRobot 会发现 `dobot_nova2` 和 `dobot_nova2_keyboard`。

只遥操不保存数据：

```powershell
cd E:\jushenzhineng\lerobot
conda run -n lerobot lerobot-teleoperate --robot.type=dobot_nova2 --robot.host=192.168.5.1 --robot.startup_on_connect=true --robot.control_mode=movl --robot.speed_factor=5 --robot.max_xyz_step_mm=1 --robot.max_rot_step_deg=0.5 --teleop.type=dobot_nova2_keyboard --teleop.xyz_step=1 --teleop.rot_step=0.5
```

记录 SmolVLA 所需数据时，需要加相机和 dataset 参数，例如：

```powershell
conda run -n lerobot lerobot-record --robot.type=dobot_nova2 --robot.host=192.168.5.1 --robot.startup_on_connect=true --robot.control_mode=movl --robot.speed_factor=5 --robot.max_xyz_step_mm=1 --robot.max_rot_step_deg=0.5 --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --teleop.type=dobot_nova2_keyboard --dataset.repo_id=你的账号/nova2_task --dataset.num_episodes=20 --dataset.fps=10
```

## Action / Observation 设计

Observation：

```text
joint_1.pos ... joint_6.pos
tcp.x, tcp.y, tcp.z, tcp.rx, tcp.ry, tcp.rz
gripper.pos
camera images（如果配置 cameras）
```

Action：

```text
delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper
```

这里使用 TCP 增量动作，是为了让键盘遥操、数据采集和后续 SmolVLA 推理都先运行在小步控制空间里，降低策略输出异常时的大幅跳动风险。

## 后续复现 SmolVLA 的路线

1. 先用 `teleop_console` 确认 Nova2 连接、运动方向、速度和急停。
2. 用 `lerobot-record` 采集包含至少一个相机视角的数据集。
3. 检查数据集 action/observation 字段稳定，FPS 不掉帧。
4. 使用 LeRobot 的 SmolVLA 训练命令训练策略。
5. 用 `lerobot-record --policy.path=...` 或对应 inference 脚本在 Nova2 上闭环复现。

## 离线测试

不连接机械臂也可以跑：

```powershell
cd E:\jushenzhineng\lerobot\lerobot_robot_dobot_nova2
conda run -n lerobot python -m pytest tests
```
