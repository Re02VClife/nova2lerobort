"""Read parquet files on Linux host to get actual episode counts and data quality."""
import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.0.130', username='jjk', password='ym123456', timeout=10, look_for_keys=False, allow_agent=False)

def run(cmd):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=20)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if err.strip():
        print("STDERR:", err.strip()[:500])
    return out

BASE = "/home/jjk/dobot_nova2_project/datasets"

# Build a single script to run on remote
script = r'''
import pyarrow.parquet as pq
import json, glob
from collections import Counter

base = "/home/jjk/dobot_nova2_project/datasets"

for ds_name in [
    "nova2_yellow_block_train32_gripper_abs",
    "nova2_yellow_block_val8_gripper_abs",
    "nova2_yellow_block_40_gripper_abs",
]:
    path = f"{base}/{ds_name}"
    print(f"\n{'='*60}")
    print(f"DATASET: {ds_name}")
    print(f"{'='*60}")

    for pf in sorted(glob.glob(f"{path}/data/chunk-*/file-*.parquet")):
        t = pq.read_table(pf)
        print(f"  Data parquet: {t.num_rows} rows")
        print(f"  Columns: {t.column_names}")
        if 'episode_index' in t.column_names:
            epis = t.column('episode_index').to_pylist()
            unique = sorted(set(epis))
            print(f"  Episodes: {len(unique)}, indexes: {unique[0]} to {unique[-1]}")
            cnt = Counter(epis)
            for ep in unique[:3]:
                print(f"    ep_{ep}: {cnt[ep]} frames")
            if len(unique) > 6:
                print(f"    ...")
            for ep in unique[-3:]:
                print(f"    ep_{ep}: {cnt[ep]} frames")

        # Check gripper values
        if 'action' in t.column_names:
            action_col = t.column('action').to_pylist()
            gripper_vals = [a[6] for a in action_col if len(a) > 6]
            if gripper_vals:
                unique_g = sorted(set(gripper_vals))
                print(f"  Gripper values: {unique_g[:15]}")
                non_zero = sum(1 for g in gripper_vals if abs(g) > 0.01)
                total = len(gripper_vals)
                print(f"  Non-zero gripper actions: {non_zero}/{total} ({100*non_zero/total:.1f}%)")

print(f"\n{'='*60}")
print("FPS TEST DATASETS")
print(f"{'='*60}")
for ds in sorted(glob.glob(f"{base}/nova2_fps_test*")):
    name = ds.split("/")[-1]
    try:
        info = json.load(open(f"{ds}/meta/info.json"))
        fps = info.get('fps', '?')
        print(f"{name}: fps={fps}")
        for pf in sorted(glob.glob(f"{ds}/data/chunk-*/file-*.parquet")):
            t = pq.read_table(pf)
            print(f"  rows={t.num_rows}, cols={t.column_names}")
    except Exception as e:
        print(f"{name}: error={e}")

print(f"\n{'='*60}")
print("ALL DATASETS QUICK SUMMARY")
print(f"{'='*60}")
for ds in sorted(glob.glob(f"{base}/nova2_*")):
    name = ds.split("/")[-1]
    total_frames = 0
    total_episodes = 0
    for pf in sorted(glob.glob(f"{ds}/data/chunk-*/file-*.parquet")):
        t = pq.read_table(pf)
        total_frames += t.num_rows
        if 'episode_index' in t.column_names:
            epis = set(t.column('episode_index').to_pylist())
            total_episodes = max(total_episodes, len(epis))
    size = sum(int(pq.read_metadata(pf).serialized_size) for pf in glob.glob(f"{ds}/data/chunk-*/file-*.parquet"))
    # Get fps from info
    fps = "?"
    try:
        info = json.load(open(f"{ds}/meta/info.json"))
        fps = info.get('fps', '?')
    except:
        pass
    if total_frames > 0:
        print(f"  {name}: {total_frames} frames, {total_episodes} episodes, fps={fps}")

print(f"\n{'='*60}")
print("STARVLA RESULTS")
print(f"{'='*60}")
import os
for root, dirs, files in os.walk(os.path.expanduser("~/starVLA/results")):
    for f in sorted(files):
        fp = os.path.join(root, f)
        sz = os.path.getsize(fp)
        print(f"  {fp} ({sz} bytes)")
    # Only top 2 levels
    if root.count(os.sep) - os.path.expanduser("~/starVLA/results").count(os.sep) >= 2:
        dirs.clear()
'''

print("Uploading and running analysis script on remote...")
# Write to temp file and execute
run(f"cat > /tmp/analyze_datasets.py << 'ENDSCRIPT'\n{script}\nENDSCRIPT")
result = run("source /home/jjk/miniconda3/etc/profile.d/conda.sh && conda run --no-capture-output -n nova2_lerobot python3 /tmp/analyze_datasets.py")
print(result)

print("\nDone.")
client.close()
