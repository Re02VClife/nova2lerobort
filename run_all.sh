#!/bin/bash
# Download SmolVLA models and launch training via SSH key
set -euo pipefail

SSH_KEY="$HOME/.ssh/id_ed25519_nova2"
SERVER="jjk@192.168.0.130"
SSH="ssh -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

echo "=== Step 1: Download SmolVLA models ==="
$SSH $SERVER 'bash -s' << 'SCRIPT'
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
source /home/jjk/miniconda3/etc/profile.d/conda.sh
MODEL_DIR="$HOME/桌面/Project/lerobot-main/models"
mkdir -p "$MODEL_DIR"

echo "--- Download SmolVLM2-500M-Video-Instruct ---"
if [ -f "$MODEL_DIR/SmolVLM2-500M-Video-Instruct/config.json" ]; then
    echo "Already exists."
else
    conda run -n nova2_lerobot hf download \
        HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
        --local-dir "$MODEL_DIR/SmolVLM2-500M-Video-Instruct"
    echo "Done."
fi

echo "--- Download smolvla_base ---"
if [ -f "$MODEL_DIR/smolvla_base/config.json" ]; then
    echo "Already exists."
else
    conda run -n nova2_lerobot hf download \
        lerobot/smolvla_base \
        --local-dir "$MODEL_DIR/smolvla_base"
    echo "Done."
fi

echo "Models:"
ls -la "$MODEL_DIR/"
SCRIPT

echo ""
echo "=== Step 2: Verify models ==="
$SSH $SERVER "ls \$HOME/桌面/Project/lerobot-main/models/SmolVLM2-500M-Video-Instruct/config.json && echo 'SmolVLM2 OK' || echo 'SmolVLM2 MISSING'"
$SSH $SERVER "ls \$HOME/桌面/Project/lerobot-main/models/smolvla_base/config.json && echo 'smolvla_base OK' || echo 'smolvla_base MISSING'"

echo ""
echo "=== Step 3: Launch SmolVLA training ==="
$SSH $SERVER 'bash -s' << 'TRAINSCRIPT'
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
source /home/jjk/miniconda3/etc/profile.d/conda.sh

LEROBOT="$HOME/桌面/Project/lerobot-main"
MODEL_DIR="$LEROBOT/models"
DATASET_ROOT="$HOME/dobot_nova2_project/datasets/nova2_gamepad_6dof_raw_20260717_092054"
OUTPUT_DIR="$HOME/dobot_nova2_project/outputs/train/nova2_smolvla_gamepad"
JOB_NAME="nova2_smolvla_gamepad"

mkdir -p "$HOME/dobot_nova2_project/outputs"

# Backup old output
test -e "$OUTPUT_DIR" && mv "$OUTPUT_DIR" "${OUTPUT_DIR}.OLD_$(date +%Y%m%d_%H%M%S)" || true

echo "Output: $OUTPUT_DIR"
echo "Dataset: $DATASET_ROOT"

cd "$LEROBOT"
nohup conda run --no-capture-output -n nova2_lerobot python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/nova2_gamepad_6dof_raw_20260717_092054 \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=smolvla \
  --policy.pretrained_path="$MODEL_DIR/smolvla_base" \
  --policy.vlm_model_name="$MODEL_DIR/SmolVLM2-500M-Video-Instruct" \
  --policy.repo_id="local/$JOB_NAME" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --policy.chunk_size=50 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --batch_size=2 \
  --steps=20000 \
  --save_freq=10000 \
  --log_freq=100 \
  --eval_freq=10000 \
  --wandb.enable=false \
  > "$HOME/dobot_nova2_project/outputs/${JOB_NAME}_train.log" 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
sleep 5
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "Training is running!"
    echo "Monitor: tail -f ~/dobot_nova2_project/outputs/${JOB_NAME}_train.log"
else
    echo "Training failed to start. Check log:"
    tail -30 "$HOME/dobot_nova2_project/outputs/${JOB_NAME}_train.log"
fi
TRAINSCRIPT

echo ""
echo "=== ALL DONE ==="
echo "Monitor training:"
echo "  ssh -i ~/.ssh/id_ed25519_nova2 jjk@192.168.0.130"
echo "  tail -f ~/dobot_nova2_project/outputs/nova2_smolvla_gamepad_train.log"
