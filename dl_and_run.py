"""Download SmolVLA models and launch training on Linux server."""
import paramiko, time, sys

HOST = '192.168.0.130'
USER = 'jjk'
PASS = 'ym123456'
CI = 'source /home/jjk/miniconda3/etc/profile.d/conda.sh'
LEROBOT = '/home/jjk/桌面/Project/lerobot-main'
MODEL_DIR = f'{LEROBOT}/models'
DATASET_ROOT = '/home/jjk/dobot_nova2_project/datasets/nova2_gamepad_6dof_raw_20260717_092054'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS, timeout=10, look_for_keys=False, allow_agent=False)
print('Connected to server.')

def run(cmd, timeout=60):
    i, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', errors='replace')
    err = e.read().decode('utf-8', errors='replace')
    if err.strip():
        print('STDERR:', err.strip()[:500])
    return out

def run_async(cmd):
    """Run command and stream output."""
    i, o, e = c.exec_command(cmd, get_pty=False)
    return i, o, e

# Step 1: Create model directory and download models
print('=' * 60)
print('STEP 1: Download SmolVLA models from HF mirror')
print('=' * 60)

# Write download script to server
dl_script = """#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
"""
dl_script += f'source {CI.split()[1]}\n'
dl_script += f"""
MODEL_DIR="{MODEL_DIR}"
mkdir -p "$MODEL_DIR"

echo "=== Download SmolVLM2-500M-Video-Instruct ==="
if [ -d "$MODEL_DIR/SmolVLM2-500M-Video-Instruct" ]; then
    echo "Already exists, skipping."
else
    conda run -n nova2_lerobot hf download \
        HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
        --local-dir "$MODEL_DIR/SmolVLM2-500M-Video-Instruct"
    echo "SmolVLM2 downloaded."
fi

echo ""
echo "=== Download smolvla_base ==="
if [ -d "$MODEL_DIR/smolvla_base" ]; then
    echo "Already exists, skipping."
else
    conda run -n nova2_lerobot hf download \
        lerobot/smolvla_base \
        --local-dir "$MODEL_DIR/smolvla_base"
    echo "smolvla_base downloaded."
fi

echo ""
echo "=== Models directory ==="
ls -la "$MODEL_DIR/" 2>/dev/null || echo "empty"
"""

i, o, e = c.exec_command('cat > /tmp/dl_models.sh', timeout=10)
i.write(dl_script)
i.close()
o.read(); e.read()
run('chmod +x /tmp/dl_models.sh')

print('Download script written. Running download (may take 5-15 min)...')
print()

# Stream the download output
i, o, e = c.exec_command('bash /tmp/dl_models.sh 2>&1', timeout=900)
start = time.time()
while not o.channel.exit_status_ready():
    if o.channel.recv_ready():
        chunk = o.channel.recv(8192).decode('utf-8', errors='replace')
        print(chunk, end='', flush=True)
    if time.time() - start > 900:
        print('\n[TIMEOUT after 15min]')
        break
    time.sleep(0.2)
# Get any remaining
try:
    remaining = o.read().decode('utf-8', errors='replace')
    if remaining:
        print(remaining)
except:
    pass
err_text = e.read().decode('utf-8', errors='replace')
if err_text.strip():
    print('STDERR:', err_text[-1000:])

print()
print('Download phase complete.')
print()

# Step 2: Verify models downloaded
print('=' * 60)
print('STEP 2: Verify models')
print('=' * 60)
for model_name in ['SmolVLM2-500M-Video-Instruct', 'smolvla_base']:
    path = f'{MODEL_DIR}/{model_name}'
    out = run(f'ls {path}/config.json 2>/dev/null && echo "OK" || echo "MISSING"')
    print(f'  {model_name}: {out.strip()}')

print()

# Step 3: Launch SmolVLA training
print('=' * 60)
print('STEP 3: Launch SmolVLA training')
print('=' * 60)

OUTPUT_DIR = '/home/jjk/dobot_nova2_project/outputs/train/nova2_smolvla_gamepad'
JOB_NAME = 'nova2_smolvla_gamepad'

# Build training command
train_cmd = f"""cd {LEROBOT} && \
{CI} && \
export HF_ENDPOINT=https://hf-mirror.com && \
conda run --no-capture-output -n nova2_lerobot python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/nova2_gamepad_6dof_raw_20260717_092054 \
  --dataset.root={DATASET_ROOT} \
  --policy.type=smolvla \
  --policy.pretrained_path={MODEL_DIR}/smolvla_base \
  --policy.vlm_model_name={MODEL_DIR}/SmolVLM2-500M-Video-Instruct \
  --policy.repo_id=local/{JOB_NAME} \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --policy.chunk_size=50 \
  --output_dir={OUTPUT_DIR} \
  --job_name={JOB_NAME} \
  --batch_size=2 \
  --steps=20000 \
  --save_freq=10000 \
  --log_freq=100 \
  --eval_freq=10000 \
  --wandb.enable=false"""

train_script = f"""#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
{CI}

echo "Output dir: {OUTPUT_DIR}"
echo "Job: {JOB_NAME}"

# Backup old output if exists
test -e {OUTPUT_DIR} && mv {OUTPUT_DIR} {OUTPUT_DIR}.OLD_$(date +%Y%m%d_%H%M%S) || true

cd {LEROBOT}
conda run --no-capture-output -n nova2_lerobot python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/nova2_gamepad_6dof_raw_20260717_092054 \
  --dataset.root={DATASET_ROOT} \
  --policy.type=smolvla \
  --policy.pretrained_path={MODEL_DIR}/smolvla_base \
  --policy.vlm_model_name={MODEL_DIR}/SmolVLM2-500M-Video-Instruct \
  --policy.repo_id=local/{JOB_NAME} \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --policy.chunk_size=50 \
  --output_dir={OUTPUT_DIR} \
  --job_name={JOB_NAME} \
  --batch_size=2 \
  --steps=20000 \
  --save_freq=10000 \
  --log_freq=100 \
  --eval_freq=10000 \
  --wandb.enable=false
"""

# Write train script
i, o, e = c.exec_command('cat > /tmp/train_smolvla.sh', timeout=10)
i.write(train_script)
i.close()
o.read(); e.read()
run('chmod +x /tmp/train_smolvla.sh')

print('Training script written.')
print()
print('Training command:')
print(f'  policy.type=smolvla')
print(f'  pretrained_path={MODEL_DIR}/smolvla_base')
print(f'  vlm_model_name={MODEL_DIR}/SmolVLM2-500M-Video-Instruct')
print(f'  dataset={DATASET_ROOT}')
print(f'  output_dir={OUTPUT_DIR}')
print(f'  batch_size=2, steps=20000')
print()

# Launch training in background with nohup
print('Launching training in background...')
launch = f"""nohup bash /tmp/train_smolvla.sh > /home/jjk/dobot_nova2_project/outputs/nova2_smolvla_gamepad_train.log 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
sleep 5
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "Training is running."
else
    echo "Training may have failed. Check log."
    tail -30 /home/jjk/dobot_nova2_project/outputs/nova2_smolvla_gamepad_train.log
fi"""

out = run(f'mkdir -p /home/jjk/dobot_nova2_project/outputs && bash -c \'{launch}\'')
print(out)

print()
print('=' * 60)
print('Setup complete!')
print('=' * 60)
print('Monitor training:')
print('  ssh jjk@192.168.0.130')
print('  tail -f ~/dobot_nova2_project/outputs/nova2_smolvla_gamepad_train.log')
print()
print('Output checkpoints:')
print(f'  {OUTPUT_DIR}')

c.close()
