"""
一键 SmolVLA 训练脚本
用法：
    python train_smolvla_full.py

步骤：
    1. 从 HF 镜像下载 SmolVLA 预训练模型到服务器
    2. 使用 Nova2 gamepad 6DOF 数据集启动 SmolVLA 微调训练
"""
import paramiko, time

HOST, USER, PASS = '192.168.0.130', 'jjk', 'ym123456'
CI = 'source /home/jjk/miniconda3/etc/profile.d/conda.sh'
LEROBOT = '/home/jjk/桌面/Project/lerobot-main'
MODEL_DIR = f'{LEROBOT}/models'
DATASET = '/home/jjk/dobot_nova2_project/datasets/nova2_gamepad_6dof_raw_20260717_092054'
OUTPUT_DIR = '/home/jjk/dobot_nova2_project/outputs/train/nova2_smolvla_gamepad'
JOB = 'nova2_smolvla_gamepad'

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=10,
              look_for_keys=False, allow_agent=False)
    return c

def run(c, cmd, timeout=60):
    i, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', errors='replace')
    err = e.read().decode('utf-8', errors='replace')
    return out, err

def stream(c, cmd, timeout=900):
    """Run with streaming output."""
    i, o, e = c.exec_command(cmd, timeout=timeout)
    start = time.time()
    while not o.channel.exit_status_ready():
        if o.channel.recv_ready():
            print(o.channel.recv(8192).decode('utf-8', errors='replace'), end='', flush=True)
        if time.time() - start > timeout:
            print('\n[TIMEOUT]')
            break
        time.sleep(0.2)
    rest = o.read().decode('utf-8', errors='replace')
    if rest: print(rest)
    err = e.read().decode('utf-8', errors='replace')
    if err.strip(): print('STDERR:', err.strip()[-500:])

# ================================================================
print('=' * 60)
print('SmolVLA Training Pipeline')
print('=' * 60)
print(f'Server:  {HOST}')
print(f'Dataset: {DATASET}')
print(f'Models:  {MODEL_DIR}')
print(f'Output:  {OUTPUT_DIR}')
print()

c = connect()
print('Connected.')

# ---- Step 1: Download models -----------------------------------
print('=' * 60)
print('STEP 1: Download SmolVLA models from HF Mirror')
print('=' * 60)

dl_script = f'''#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
{CI}
MD="{MODEL_DIR}"
mkdir -p "$MD"

echo ">>> Downloading SmolVLM2-500M-Video-Instruct..."
if [ -f "$MD/SmolVLM2-500M-Video-Instruct/config.json" ]; then
    echo "Already exists, skip."
else
    conda run -n nova2_lerobot hf download \
        HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
        --local-dir "$MD/SmolVLM2-500M-Video-Instruct"
    echo "SmolVLM2 done."
fi

echo ">>> Downloading smolvla_base..."
if [ -f "$MD/smolvla_base/config.json" ]; then
    echo "Already exists, skip."
else
    conda run -n nova2_lerobot hf download \
        lerobot/smolvla_base \
        --local-dir "$MD/smolvla_base"
    echo "smolvla_base done."
fi

echo ""
echo "Models downloaded:"
ls -la "$MD/"
'''

i, o, e = c.exec_command('cat > /tmp/dl_smolvla.sh', timeout=10)
i.write(dl_script); i.close()
o.read(); e.read()
print('Download script written. Running (5-15 min)...')
print()

i, o, e = c.exec_command('bash /tmp/dl_smolvla.sh 2>&1', timeout=900)
start = time.time()
while not o.channel.exit_status_ready():
    if o.channel.recv_ready():
        print(o.channel.recv(8192).decode('utf-8', errors='replace'), end='', flush=True)
    if time.time() - start > 900: break
    time.sleep(0.2)
rest = o.read().decode('utf-8', errors='replace')
if rest: print(rest)
err = e.read().decode('utf-8', errors='replace')
if err.strip(): print('STDERR:', err.strip()[-500:])

# ---- Step 2: Verify models ------------------------------------
print()
print('=' * 60)
print('STEP 2: Verify models downloaded')
print('=' * 60)
for name in ['SmolVLM2-500M-Video-Instruct', 'smolvla_base']:
    out, _ = run(c, f'test -f {MODEL_DIR}/{name}/config.json && echo "OK" || echo "MISSING"')
    print(f'  {name}: {out.strip()}')

# ---- Step 3: Launch training ----------------------------------
print()
print('=' * 60)
print('STEP 3: Launch SmolVLA Training')
print('=' * 60)

train_script = f'''#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
{CI}

LE="{LEROBOT}"
MD="{MODEL_DIR}"
DS="{DATASET}"
OUT="{OUTPUT_DIR}"
JOB="{JOB}"

mkdir -p /home/jjk/dobot_nova2_project/outputs

# Backup old output
test -e "$OUT" && mv "$OUT" "${{OUT}}.OLD_$(date +%Y%m%d_%H%M%S)" || true

echo "Launching SmolVLA training..."
echo "  Pretrained: $MD/smolvla_base"
echo "  VLM:        $MD/SmolVLM2-500M-Video-Instruct"
echo "  Dataset:    $DS"
echo "  Output:     $OUT"
echo "  Device:     cuda"
echo "  Steps:      20000 (save every 10000)"
echo "  Batch size: 2"
echo ""

cd "$LE"
conda run --no-capture-output -n nova2_lerobot python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/nova2_gamepad_6dof_raw_20260717_092054 \
  --dataset.root="$DS" \
  --policy.type=smolvla \
  --policy.pretrained_path="$MD/smolvla_base" \
  --policy.vlm_model_name="$MD/SmolVLM2-500M-Video-Instruct" \
  --policy.repo_id="local/$JOB" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --policy.chunk_size=50 \
  --output_dir="$OUT" \
  --job_name="$JOB" \
  --batch_size=2 \
  --steps=20000 \
  --save_freq=10000 \
  --log_freq=100 \
  --eval_freq=10000 \
  --wandb.enable=false
'''

i, o, e = c.exec_command('cat > /tmp/train_smolvla.sh', timeout=10)
i.write(train_script); i.close()
o.read(); e.read()

print('Training script written. Launching in background...')
print()

# Launch with nohup
launch_cmd = '''nohup bash /tmp/train_smolvla.sh > /home/jjk/dobot_nova2_project/outputs/nova2_smolvla_gamepad_train.log 2>&1 &
TRAIN_PID=$!
echo "PID: $TRAIN_PID"
sleep 8
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "TRAINING_RUNNING"
    echo "PID=$TRAIN_PID"
else
    echo "TRAINING_FAILED"
    tail -30 /home/jjk/dobot_nova2_project/outputs/nova2_smolvla_gamepad_train.log
fi'''

out, err = run(c, launch_cmd, timeout=30)
print(out)
if err: print('ERR:', err.strip()[-300:])

# ---- Summary --------------------------------------------------
c.close()
print()
print('=' * 60)
print('SETUP COMPLETE')
print('=' * 60)
print()
print('Monitor training:')
print('  ssh jjk@192.168.0.130')
print('  tail -f ~/dobot_nova2_project/outputs/nova2_smolvla_gamepad_train.log')
print()
print('Checkpoints will be at:')
print(f'  {OUTPUT_DIR}/checkpoints/')
print()
print('Training config:')
print('  policy:      smolvla')
print('  dataset:     40 episodes, wrist+front cameras, 20fps, 31K frames')
print('  batch_size:  2')
print('  steps:       20000')
print('  save_freq:   10000')
print('  device:      cuda (RTX 4090D 24GB)')
