"""Check datasets on Linux host via paramiko."""
import paramiko, json

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.0.130', username='jjk', password='ym123456', timeout=10, look_for_keys=False, allow_agent=False)
print('Connected to Linux host\n')

def run(cmd, timeout=15):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if err.strip():
        print("[stderr]", err.strip()[:200])
    return out

BASE = "/home/jjk/dobot_nova2_project/datasets"

# Helper: list episodes from a dataset
def show_episodes(name):
    path = f"{BASE}/{name}"
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    # Read info.json
    info_raw = run(f"cat {path}/meta/info.json 2>/dev/null")
    if info_raw.strip():
        try:
            info = json.loads(info_raw.strip())
            print(f"  fps={info.get('fps','?')}, codebase={info.get('codebase_version','?')}")
            # camera info
            for cam in ['observation.images.wrist', 'observation.images.front']:
                c = info.get('features', {}).get(cam, {})
                shape = c.get('shape', [])
                v = c.get('info', {})
                print(f"  {cam.split('.')[-1]}: {shape}, codec={v.get('video.codec','?')}, video_fps={v.get('video.fps','?')}")
        except Exception as e:
            print(f"  [json parse error: {e}]")

    # Episode files
    epis = run(f"ls {path}/meta/episodes/episode_*.json 2>/dev/null | wc -l").strip()
    print(f"  Total episodes: {epis}")

    # Episode details
    epi_data = run(f"for f in {path}/meta/episodes/episode_*.json; do python3 -c 'import json; d=json.load(open(\\\"$f\\\")); print(\"  ep_\"+str(d.get(\"episode_index\",\"?\"))+\": frames=\"+str(d.get(\"length\",\"?\"))+\", tasks=\"+str(d.get(\"tasks\",\"?\")))'; done")
    if epi_data.strip():
        print(epi_data.strip())

    # Size
    size = run(f"du -sh {path} 2>/dev/null | cut -f1").strip()
    print(f"  Total size: {size}")

# 1. Key datasets
for ds in [
    "nova2_yellow_block_train32_gripper_abs",
    "nova2_yellow_block_val8_gripper_abs",
    "nova2_yellow_block_40_gripper_abs",
    "nova2_gamepad_6dof_raw_20260717_092054",
    "nova2_fps_test_640_20260716_114228",
    "nova2_button_flow_test_20260716_151736",
]:
    show_episodes(ds)

# 2. All datasets summary
print(f"\n{'='*60}")
print("ALL DATASETS SUMMARY")
print(f"{'='*60}")
summary = run(f"""
for d in {BASE}/nova2_*; do
    name=$(basename "$d")
    epis=$(ls "$d"/meta/episodes/episode_*.json 2>/dev/null | wc -l)
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    fps=$(python3 -c "import json; print(json.load(open('$d/meta/info.json')).get('fps','?'))" 2>/dev/null)
    echo "  $name: size=$size, episodes=$epis, fps=$fps"
done
""")
print(summary.strip())

# 3. Video sizes for train32
print(f"\n{'='*60}")
print("VIDEO SIZES - train32")
print(f"{'='*60}")
v = run(f"du -sh {BASE}/nova2_yellow_block_train32_gripper_abs/videos/*/ 2>/dev/null")
print(v.strip())

# 4. Quality check script
print(f"\n{'='*60}")
print("QUALITY CHECK SCRIPT")
print(f"{'='*60}")
qc = run(f"head -80 /home/jjk/dobot_nova2_project/scripts/quality_check.py 2>/dev/null")
print(qc.strip()[:3000])

# 5. VALID_EPISODES
print(f"\n{'='*60}")
print("VALID EPISODES")
print(f"{'='*60}")
ve = run(f"cat {BASE}/VALID_EPISODES_CURRENT.txt 2>/dev/null || echo 'Not found'")
print(ve.strip()[:2000])

# 6. Data directory structure
print(f"\n{'='*60}")
print("DATA DIRECTORY (parquet files)")
print(f"{'='*60}")
dd = run(f"ls -lh {BASE}/nova2_yellow_block_train32_gripper_abs/data/chunk-*/")
print(dd.strip())

# 7. Check StarVLA training state
print(f"\n{'='*60}")
print("STARVLA TRAINING CHECK")
print(f"{'='*60}")
sv = run("ls -lh ~/starVLA/ 2>/dev/null | head -20; echo '---'; ls -lh ~/starVLA/outputs/ 2>/dev/null | head -20")
print(sv.strip()[:2000])

client.close()
print("\n\nDone.")
