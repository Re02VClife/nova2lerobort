#!/usr/bin/env python3
"""
从真机数据集抽取 hover（爪开、方块可见）与 grasp（爪闭、已抓取）帧，
记录每帧 tcp 真机坐标系位姿，用于顶视相机标定。
用法（服务器）:
    conda run -n nova2_lerobot python extract_real_grasp.py \
        --dataset ~/dobot_nova2_project/datasets/nova2_yellow_block_train52_gripper_clean \
        --out /tmp/real_grasp  --n-episodes 6
"""
from __future__ import annotations
import argparse, json, os

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="/tmp/real_grasp")
    ap.add_argument("--n-episodes", type=int, default=6)
    args = ap.parse_args()

    ds = LeRobotDataset(args.dataset)
    os.makedirs(args.out, exist_ok=True)
    print(f"[ext] {args.dataset} eps={ds.num_episodes} frames={ds.num_frames}", flush=True)

    ep_idx_col = ds.hf_dataset["episode_index"]
    boundaries, cur_ep, cur_start = [], None, 0
    for i, e in enumerate(ep_idx_col):
        if cur_ep is None:
            cur_ep, cur_start = e, i
        elif e != cur_ep:
            boundaries.append((cur_start, i))
            cur_ep, cur_start = e, i
    if cur_ep is not None:
        boundaries.append((cur_start, len(ep_idx_col)))

    out = {}
    for ep in range(min(args.n_episodes, len(boundaries))):
        f0, f1 = boundaries[ep]
        # observation.state: [0:6]=joints, [6:9]=tcp.xyz, [9:12]=tcp.rpy, [12]=gripper
        zs = [ds[f]["observation.state"][8] for f in range(f0, f1)]
        grippers = [ds[f]["observation.state"][12] for f in range(f0, f1)]
        zmin = min(zs)
        grasp_f = f0 + int(np.argmin(zs))
        # hover 帧：抓取前最后一个夹爪张开(>0.7) 的帧
        hover_f = None
        for f in range(grasp_f - 1, f0 - 1, -1):
            if grippers[f - f0] > 0.7:
                hover_f = f
                break
        if hover_f is None:
            hover_f = max(f0, grasp_f - 5)

        meta = {"ep": ep, "from": int(f0), "to": int(f1), "tcp_z_min": round(float(zmin), 1)}
        for tag, f in (("hover", hover_f), ("grasp", grasp_f)):
            st = ds[f]["observation.state"]
            meta[tag] = {
                "frame": int(f),
                "tcp": [round(float(v), 1) for v in st[6:9]],      # tcp.x/y/z real mm
                "rpy": [round(float(v), 1) for v in st[9:12]],     # tcp.rx/ry/rz real deg
                "gripper": round(float(st[12]), 3),
                "joints": [round(float(v), 4) for v in st[0:6]],
            }
            img = ds[f]["observation.images.front"]
            if isinstance(img, np.ndarray):
                arr = img
            else:
                arr = img.permute(1, 2, 0).cpu().numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(args.out, f"real_ep{ep:02d}_{tag}_front.png"))
        out[f"ep{ep}"] = meta
        print(json.dumps(meta, ensure_ascii=False), flush=True)

    with open(os.path.join(args.out, "_tcp.json"), "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[ext] DONE -> {args.out}")


if __name__ == "__main__":
    main()
