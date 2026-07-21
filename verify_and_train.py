"""Verify dataset compatibility and set up isolated StarVLA training."""
import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.130', username='jjk', password='ym123456', timeout=10, look_for_keys=False, allow_agent=False)

conda_init = "source /home/jjk/miniconda3/etc/profile.d/conda.sh"

def run(cmd, timeout=20):
    i, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', errors='replace')
    err = e.read().decode('utf-8', errors='replace')
    if err.strip():
        print("STDERR:", err.strip()[:400])
    return out

# 1. Verify starvla env works
print("=" * 60)
print("1. STARVLA ENV CHECK")
print("=" * 60)
script = '''import torch; print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}, vram={torch.cuda.get_device_properties(0).total_mem//1024**3}GB")'''
cmd = f"""{conda_init} && conda run --no-capture-output -n starvla python3 -c '{script}'"""
print(run(cmd))

# 2. Verify dataset format compatibility
print("=" * 60)
print("2. DATASET FORMAT COMPATIBILITY")
print("=" * 60)
run("cat > /tmp/check_format.py << 'ENDPY'\n"
    "import pyarrow.parquet as pq\n"
    "new = '/home/jjk/dobot_nova2_project/datasets/nova2_gamepad_6dof_raw_20260717_092054/data/chunk-000/file-000.parquet'\n"
    "old = '/home/jjk/dobot_nova2_project/datasets/nova2_yellow_block_train32_gripper_abs/data/chunk-000/file-000.parquet'\n"
    "tn = pq.read_table(new)\n"
    "to = pq.read_table(old)\n"
    "print(f'New dataset: {tn.num_rows} rows, cols={tn.column_names}')\n"
    "print(f'Old dataset: {to.num_rows} rows, cols={to.column_names}')\n"
    "an = tn.column('action').to_pylist()\n"
    "ao = to.column('action').to_pylist()\n"
    "sn = tn.column('observation.state').to_pylist()\n"
    "so = to.column('observation.state').to_pylist()\n"
    "print(f'New action[0]:  {an[0]}')\n"
    "print(f'Old action[0]:  {ao[0]}')\n"
    "print(f'New state[0]:   {sn[0][:6]}...')\n"
    "print(f'Old state[0]:   {so[0][:6]}...')\n"
    "print(f'Action dim match: {len(an[0]) == len(ao[0])}')\n"
    "print(f'State dim match: {len(sn[0]) == len(so[0])}')\n"
    "en = sorted(set(tn.column('episode_index').to_pylist()))\n"
    "print(f'New episodes: {en[0]}..{en[-1]} ({len(en)} total)')\n"
    "eo = sorted(set(to.column('episode_index').to_pylist()))\n"
    "print(f'Old episodes: {eo[0]}..{eo[-1]} ({len(eo)} total)')\n"
    "ENDPY")
cmd = f"{conda_init} && conda run --no-capture-output -n nova2_lerobot python3 /tmp/check_format.py"
print(run(cmd))

# 3. Disk space
print("=" * 60)
print("3. DISK SPACE")
print("=" * 60)
print(run("df -h /home/jjk/ 2>/dev/null"))

# 4. GPU state
print("=" * 60)
print("4. GPU STATE")
print("=" * 60)
print(run("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader"))

# 5. Check if starvla training script exists
print("=" * 60)
print("5. STARVLA TRAINING SCRIPT")
print("=" * 60)
print(run("ls -lh ~/starVLA/starVLA/training/train_starvla.py 2>/dev/null; ls ~/starVLA/starVLA/config/deepseeds/deepspeed_zero2.yaml 2>/dev/null"))

# 6. Check what processes are using GPU
print("=" * 60)
print("6. GPU PROCESSES")
print("=" * 60)
print(run("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null"))

c.close()
print("\nDone.")
