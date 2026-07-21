#!/usr/bin/env python3
"""
Nova2 数据质检脚本
对录制完成的 episode 进行结构质检和基础分析

用法：
    conda run --no-capture-output -n nova2_lerobot python quality_check.py \
        --dataset /home/jjk/dobot_nova2_project/datasets/nova2_yellow_block_train30_part1

质检项：
    1. Episode 数量
    2. State/Action 维度
    3. 夹爪动作完整性（是否包含 -1/0/+1）
    4. 动作限幅（XYZ ≤ 1.0, 旋转 ≤ 0.5）
    5. 数值有效性（NaN/Inf 检测）
    6. TCP Y 边界（≥ -560mm）
    7. 视频文件存在性
"""

import argparse
import json
import os
import sys
import pathlib
import numpy as np
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
EXPECTED_STATE_DIM = 13   # 6关节 + 6TCP + 1夹爪
EXPECTED_ACTION_DIM = 7   # delta_x/y/z + delta_rx/ry/rz + gripper
XYZ_MAX = 1.0              # mm/frame
ROT_MAX = 0.5              # deg/frame
Y_MIN = -560               # mm


def load_dataset(root: str) -> dict:
    """加载数据集元信息和数据"""
    root = pathlib.Path(root)
    info_path = root / "meta" / "info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"元数据文件不存在: {info_path}")

    with open(info_path) as f:
        meta = json.load(f)

    episodes = {}
    data_dir = root / "data"
    if data_dir.exists():
        for npz_file in sorted(data_dir.glob("episode_*.npz")):
            data = np.load(npz_file)
            episodes[npz_file.name] = {
                "observations": data["observations"],
                "actions": data["actions"],
            }

    videos = list((root / "videos").rglob("*")) if (root / "videos").exists() else []

    return {"meta": meta, "episodes": episodes, "videos": videos}


def check_episode_count(meta: dict, episodes: dict) -> list:
    """第1项：检查 episode 数量"""
    issues = []
    expected = meta.get("total_episodes", 0)
    actual = len(episodes)
    if expected != actual:
        issues.append({
            "level": "ERROR",
            "item": "Episode数量",
            "detail": f"元数据声称 {expected} 条，实际数据文件 {actual} 条",
        })
    else:
        issues.append({
            "level": "OK",
            "item": "Episode数量",
            "detail": f"{actual} 条，与元数据一致",
        })
    return issues


def check_dims(episodes: dict) -> list:
    """第2项：检查 state/action 维度"""
    issues = []
    for name, ep in episodes.items():
        obs = ep["observations"]
        act = ep["actions"]

        if obs.ndim != 2:
            issues.append({
                "level": "ERROR",
                "item": "State维度",
                "detail": f"{name}: observations 不是2维数组 (shape={obs.shape})",
            })
        elif obs.shape[1] != EXPECTED_STATE_DIM:
            issues.append({
                "level": "ERROR",
                "item": "State维度",
                "detail": f"{name}: state 维度={obs.shape[1]}, 期望{EXPECTED_STATE_DIM}",
            })

        if act.ndim != 2:
            issues.append({
                "level": "ERROR",
                "item": "Action维度",
                "detail": f"{name}: actions 不是2维数组 (shape={act.shape})",
            })
        elif act.shape[1] != EXPECTED_ACTION_DIM:
            issues.append({
                "level": "ERROR",
                "item": "Action维度",
                "detail": f"{name}: action 维度={act.shape[1]}, 期望{EXPECTED_ACTION_DIM}",
            })

    if not issues:
        issues.append({
            "level": "OK",
            "item": "State/Action维度",
            "detail": f"所有 episode: state={EXPECTED_STATE_DIM}维, action={EXPECTED_ACTION_DIM}维",
        })
    return issues


def check_gripper_actions(episodes: dict) -> list:
    """第3项：检查夹爪动作是否包含 -1/0/+1"""
    issues = []
    for name, ep in episodes.items():
        actions = ep["actions"]
        gripper_actions = actions[:, -1]  # 最后一维是 gripper

        has_close = np.any(gripper_actions < -0.5)     # -1: 闭合
        has_open = np.any(gripper_actions > 0.5)        # +1: 张开

        if not has_close:
            issues.append({
                "level": "FAIL",
                "item": "夹爪动作",
                "detail": f"{name}: 缺少闭合动作 (-1)，gripper_actions 范围=[{gripper_actions.min():.2f}, {gripper_actions.max():.2f}]",
            })
        if not has_open:
            issues.append({
                "level": "FAIL",
                "item": "夹爪动作",
                "detail": f"{name}: 缺少张开动作 (+1)，gripper_actions 范围=[{gripper_actions.min():.2f}, {gripper_actions.max():.2f}]",
            })

    passing = sum(1 for name, ep in episodes.items()
                  if np.any(ep["actions"][:, -1] < -0.5) and np.any(ep["actions"][:, -1] > 0.5))
    issues.append({
        "level": "OK" if passing == len(episodes) else "WARN",
        "item": "夹爪动作汇总",
        "detail": f"{passing}/{len(episodes)} 条具备完整抓取-释放动作",
    })
    return issues


def check_action_limits(episodes: dict) -> list:
    """第4项：检查动作限幅"""
    issues = []
    for name, ep in episodes.items():
        actions = ep["actions"]
        xyz = np.abs(actions[:, :3])
        rot = np.abs(actions[:, 3:6])

        xyz_violations = np.sum(xyz > XYZ_MAX)
        rot_violations = np.sum(rot > ROT_MAX)

        if xyz_violations > 0:
            # 忽略前几帧（初始化瞬间可能有跳变）
            xyz_check = np.abs(actions[5:, :3])
            xyz_violations_adj = np.sum(xyz_check > XYZ_MAX)
            if xyz_violations_adj > 0:
                issues.append({
                    "level": "WARN",
                    "item": "动作限幅-XYZ",
                    "detail": f"{name}: {xyz_violations_adj} 帧超 {XYZ_MAX}mm (跳过前5帧)",
                })

        if rot_violations > 0:
            rot_check = np.abs(actions[5:, 3:6])
            rot_violations_adj = np.sum(rot_check > ROT_MAX)
            if rot_violations_adj > 0:
                issues.append({
                    "level": "WARN",
                    "item": "动作限幅-旋转",
                    "detail": f"{name}: {rot_violations_adj} 帧超 {ROT_MAX}° (跳过前5帧)",
                })

    if not issues:
        issues.append({
            "level": "OK",
            "item": "动作限幅",
            "detail": f"所有帧 XYZ ≤ {XYZ_MAX}mm, 旋转 ≤ {ROT_MAX}°",
        })
    return issues


def check_nan_inf(episodes: dict) -> list:
    """第5项：检查 NaN/Inf"""
    issues = []
    for name, ep in episodes.items():
        obs = ep["observations"]
        act = ep["actions"]

        nan_count_obs = np.sum(np.isnan(obs))
        inf_count_obs = np.sum(np.isinf(obs))
        nan_count_act = np.sum(np.isnan(act))
        inf_count_act = np.sum(np.isinf(act))

        total_nan_inf = nan_count_obs + inf_count_obs + nan_count_act + inf_count_act
        if total_nan_inf > 0:
            issues.append({
                "level": "FAIL",
                "item": "NaN/Inf",
                "detail": f"{name}: NaN={nan_count_obs + nan_count_act}, Inf={inf_count_obs + inf_count_act} — 禁止训练!",
            })

    if not issues:
        issues.append({
            "level": "OK",
            "item": "NaN/Inf",
            "detail": "所有 episode 数值有效",
        })
    return issues


def check_tcp_y(episodes: dict) -> list:
    """第6项：检查 TCP Y 边界"""
    issues = []
    for name, ep in episodes.items():
        obs = ep["observations"]
        # observations: [6 joints, 6 tcp, 1 gripper]
        # tcp_y = index 7 (0-indexed: joints[0:6], tcp[0:6], gripper)
        tcp_y = obs[:, 7]  # TCP Y

        y_min_actual = tcp_y.min()
        y_violations = np.sum(tcp_y < Y_MIN)

        if y_violations > 0:
            issues.append({
                "level": "FAIL",
                "item": "TCP Y边界",
                "detail": f"{name}: Y_min={y_min_actual:.2f} < {Y_MIN}mm, {y_violations} 帧越界 — 安全异常!",
            })

    if not issues:
        issues.append({
            "level": "OK",
            "item": "TCP Y边界",
            "detail": f"所有帧 TCP Y ≥ {Y_MIN}mm",
        })
    return issues


def check_videos(root: str, episodes: dict) -> list:
    """第7项：检查视频文件"""
    issues = []
    videos_dir = pathlib.Path(root) / "videos"

    if not videos_dir.exists():
        issues.append({
            "level": "WARN",
            "item": "视频文件",
            "detail": "videos 目录不存在",
        })
        return issues

    video_files = list(videos_dir.rglob("*"))
    video_count = len([v for v in video_files if v.suffix in ('.mp4', '.avi', '.mkv', '.webm')])

    if video_count == 0:
        issues.append({
            "level": "WARN",
            "item": "视频文件",
            "detail": "未找到视频文件",
        })
    else:
        issues.append({
            "level": "OK",
            "item": "视频文件",
            "detail": f"找到 {video_count} 个视频文件",
        })
    return issues


def print_summary(all_issues: list):
    """打印质检汇总"""
    levels = defaultdict(int)
    for issue in all_issues:
        levels[issue["level"]] += 1

    print("\n" + "=" * 60)
    print("  质检汇总")
    print("=" * 60)
    print(f"  OK:    {levels.get('OK', 0)}")
    print(f"  WARN:  {levels.get('WARN', 0)}")
    print(f"  FAIL:  {levels.get('FAIL', 0)}")
    print(f"  ERROR: {levels.get('ERROR', 0)}")

    if levels.get('FAIL', 0) > 0 or levels.get('ERROR', 0) > 0:
        print("\n  ❌ 存在不合格项，需处理后再用于训练")
    elif levels.get('WARN', 0) > 0:
        print("\n  ⚠ 存在警告项，建议检查后确认")
    else:
        print("\n  ✅ 全部质检通过，可用于训练")


def main():
    parser = argparse.ArgumentParser(description="Nova2 数据质检")
    parser.add_argument("--dataset", "-d", required=True, help="数据集根目录路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="逐条显示详情")
    args = parser.parse_args()

    root = args.dataset

    print("=" * 60)
    print("  Nova2 数据质检")
    print(f"  数据集: {root}")
    print("=" * 60)

    # 加载数据
    try:
        ds = load_dataset(root)
    except Exception as e:
        print(f"\n[致命错误] 无法加载数据集: {e}")
        sys.exit(1)

    print(f"\n加载完成: {len(ds['episodes'])} episodes, {len(ds['videos'])} 视频文件")

    # 逐项检查
    all_issues = []
    checks = [
        ("1/7 Episode数量", check_episode_count),
        ("2/7 State/Action维度", check_dims),
        ("3/7 夹爪动作", check_gripper_actions),
        ("4/7 动作限幅", check_action_limits),
        ("5/7 NaN/Inf", check_nan_inf),
        ("6/7 TCP Y边界", check_tcp_y),
        ("7/7 视频文件", lambda eps: check_videos(root, eps)),
    ]

    for label, check_fn in checks:
        print(f"\n[{label}]")
        results = check_fn(ds["episodes"])
        all_issues.extend(results)
        for r in results:
            icon = {"OK": "  ✓", "WARN": "  ⚠", "FAIL": "  ✗", "ERROR": "  ✗"}
            print(f"{icon.get(r['level'], '  ?')} [{r['level']}] {r['detail']}")

    # 汇总
    print_summary(all_issues)

    # 返回码
    has_fail = any(i["level"] in ("FAIL", "ERROR") for i in all_issues)
    sys.exit(1 if has_fail else 0)


if __name__ == "__main__":
    main()
