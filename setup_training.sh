#!/usr/bin/env bash
# Isolated StarVLA training setup script
# Run on Linux host: bash ~/setup_training.sh
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

CONDA_BASE="/home/jjk/miniconda3"
CONDA_INIT="source $CONDA_BASE/etc/profile.d/conda.sh"
LEROBOT_ROOT="/home/jjk/桌面/Project/lerobot-main"
STARVLA_ROOT="/home/jjk/starVLA"
DATA_ROOT="/home/jjk/dobot_nova2_project/datasets"
CONDA_LEROBOT="nova2_lerobot"
CONDA_STARVLA="starvla"

SRC="$DATA_ROOT/nova2_gamepad_6dof_raw_20260717_092054"
WORK="$DATA_ROOT/.gamepad40_split_workspace"
TRAIN_DST="$DATA_ROOT/nova2_gamepad_train32_20260717"
VAL_DST="$DATA_ROOT/nova2_gamepad_val8_20260717"
RUN_ID="nova2_gamepad_train32_isolated"

echo "============================================"
echo "StarVLA Isolated Training Setup"
echo "Run ID: $RUN_ID"
echo "Source: $SRC"
echo "============================================"

# STEP 1: Split dataset
echo ""
echo "[Step 1/5] Splitting dataset into train(32ep) + val(8ep)..."

if [[ -d "$TRAIN_DST" ]] && [[ -d "$VAL_DST" ]]; then
    echo "  Split already exists, skipping."
else
    # Clean up
    for dst in "$TRAIN_DST" "$VAL_DST"; do
        if [[ -e "$dst" ]]; then
            mv "$dst" "${dst}.OLD_$(date +%Y%m%d_%H%M%S)"
        fi
    done
    rm -rf "$WORK"
    mkdir -p "$WORK"

    cd "$LEROBOT_ROOT"
    eval "$CONDA_INIT"
    conda run --no-capture-output -n "$CONDA_LEROBOT" lerobot-edit-dataset \
        --repo_id "jjk/nova2_gamepad_6dof_raw_20260717_092054" \
        --root "$SRC" \
        --operation.type split \
        --operation.splits '{"train":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],"val":[32,33,34,35,36,37,38,39]}' \
        --new_root "$WORK" \
        --push_to_hub false

    mv "$WORK/train" "$TRAIN_DST"
    mv "$WORK/val" "$VAL_DST"
    rmdir "$WORK" 2>/dev/null || true
    echo "  Split complete:"
    echo "    Train: $TRAIN_DST"
    echo "    Val:   $VAL_DST"
fi

# STEP 2: Register datasets in StarVLA
echo ""
echo "[Step 2/5] Registering datasets..."

REGISTRY="$STARVLA_ROOT/examples/Nova2/train_files/data_registry/data_config.py"

# Backup registry
cp "$REGISTRY" "${REGISTRY}.before_${RUN_ID}_$(date +%Y%m%d_%H%M%S).bak"

if grep -q 'nova2_gamepad_train32_20260717' "$REGISTRY"; then
    echo "  Dataset already registered, skipping."
else
    cat >> "$REGISTRY" << 'ENDREG'

DATASET_NAMED_MIXTURES["nova2_gamepad_train32_20260717"] = [("nova2_gamepad_train32_20260717", 1.0, "dobot_nova2_delta_eef")]
DATASET_NAMED_MIXTURES["nova2_gamepad_val8_20260717"] = [("nova2_gamepad_val8_20260717", 1.0, "dobot_nova2_delta_eef")]
ENDREG
    echo "  Registered nova2_gamepad_train32_20260717"
    echo "  Registered nova2_gamepad_val8_20260717"
fi

# STEP 3: Create training config
echo ""
echo "[Step 3/5] Creating training config..."

CONFIG_DIR="$STARVLA_ROOT/examples/Nova2/train_files"
CONFIG="$CONFIG_DIR/starvla_${RUN_ID}.yaml"

if [[ -f "$CONFIG" ]]; then
    echo "  Config already exists, skipping."
else
    cat > "$CONFIG" << ENDCFG
run_id: $RUN_ID
run_root_dir: $STARVLA_ROOT/results/Checkpoints
seed: 42
trackers:
- jsonl
wandb_entity: jinhuiye
wandb_project: StarVLA_Libero
is_debug: false
framework:
  name: QwenGR00T
  qwenvl:
    base_vlm: $STARVLA_ROOT/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
    attn_implementation: flash_attention_2
    vl_hidden_dim: 2048
  dino:
    dino_backbone: dinov2_vits14
  action_model:
    action_model_type: DiT-B
    action_hidden_dim: 1024
    hidden_size: 1024
    add_pos_embed: true
    max_seq_len: 1024
    action_dim: 7
    state_dim: 13
    future_action_window_size: 15
    action_horizon: 16
    past_action_window_size: 0
    repeated_diffusion_steps: 1
    noise_beta_alpha: 1.5
    noise_beta_beta: 1.0
    noise_s: 0.999
    num_timestep_buckets: 1000
    num_inference_timesteps: 4
    num_target_vision_tokens: 32
    diffusion_model_cfg:
      cross_attention_dim: 2048
      dropout: 0.2
      final_dropout: true
      interleave_self_attention: true
      norm_type: ada_norm
      num_layers: 16
      output_dim: 1024
      positional_embeddings: null
    gripper_loss_weight: 5.0
    gripper_channel_idx: 6
  reduce_in_full_precision: true
datasets:
  vlm_data:
    dataset_py: vlm_datasets
    dataformat: llava_json
    dataset_use: asv2_conversation_en,asv2_detailed_description_en,asv2_region_captioning_en,coco_internvl_longcap_en,coco_karpathy_train_567_en,coco_negative_gpt4o_en,coco_poetry_zh,coco_rem_en_zh,cocorem_exist_yorn_en,cocotextv2_en,cocotextv2_gpt4o_en,okvqa_en,refcoco_grounding_aug_en,refcoco_grounding_en,tallyqa_coco_en,toloka_grounding_aug_en,vqav2_en,vsr_en
    eval_dataset: aokvqa_cauldron_llava_format
    data_flatten: false
    base_interval: 2
    max_pixels: 12845056
    min_pixels: 3136
    model_max_length: 2048
    model_type: qwen2.5vl
    per_device_batch_size: 3
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: $DATA_ROOT
    data_mix: nova2_gamepad_train32_20260717
    action_type: delta_ee
    CoT_prompt: Your task is {instruction}. To identify the key objects for your task. Locate their bounding boxes in [x1,y1,x2,y2] format.
    CoT_answer: bbox
    default_image_resolution:
    - 3
    - 224
    - 224
    per_device_batch_size: 1
    load_all_data_for_training: true
    obs:
    - image_0
    include_state: true
    obs_image_size:
    - 224
    - 224
    action_mode: abs
    lerobot_version: v3.0
    delete_pause_frame: false
    video_backend: torchvision_av
    num_workers: 0
trainer:
  epochs: 100
  max_train_steps: 100
  num_warmup_steps: 10
  save_interval: 100
  eval_interval: 100
  learning_rate:
    base: 0.0001
    action_model: 0.0001
  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 1.0e-06
  freeze_modules: qwen_vl_interface
  loss_scale:
    vla: 1.0
    vlm: 0.1
  max_grad_norm: 1.0
  warmup_ratio: 0.1
  weight_decay: 0.0
  logging_frequency: 1
  gradient_clipping: 1.0
  is_resume: false
  optimizer:
    betas:
    - 0.9
    - 0.95
    eps: 1.0e-08
    weight_decay: 1.0e-08
  pretrained_checkpoint: $STARVLA_ROOT/playground/Pretrained_models/Qwen2.5-VL-GR00T-Nova2-init/checkpoints/steps_30000_nova2_init_model.safetensors
  reload_modules: null
  save_format: safetensors
output_dir: $STARVLA_ROOT/results/Checkpoints/$RUN_ID
ENDCFG
    echo "  Config created: $CONFIG"
fi

# STEP 4: Verify GPU and environment
echo ""
echo "[Step 4/5] Checking GPU and environment..."

nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv,noheader

eval "$CONDA_INIT"
conda run --no-capture-output -n "$CONDA_STARVLA" python3 -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0)}')
print(f'vram_free={torch.cuda.get_device_properties(0).total_memory//1024**3 - torch.cuda.memory_allocated()//1024**3}GB')
"

# Check for existing training processes
echo ""
echo "Active training/model processes:"
pgrep -af 'train_starvla|accelerate.*train' 2>/dev/null || echo "  None (clean)"

# STEP 5: Launch training
echo ""
echo "[Step 5/5] Launching isolated training..."

OUTPUT_DIR="$STARVLA_ROOT/results/Checkpoints/$RUN_ID"
LOG="$STARVLA_ROOT/results/${RUN_ID}.log"

# Backup existing output
if [[ -e "$OUTPUT_DIR" ]]; then
    mv "$OUTPUT_DIR" "${OUTPUT_DIR}.OLD_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$STARVLA_ROOT/results"

cd "$STARVLA_ROOT"
eval "$CONDA_INIT"

# Launch with isolation
nohup bash -c "
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
source $CONDA_BASE/etc/profile.d/conda.sh
conda run --no-capture-output -n $CONDA_STARVLA \
    accelerate launch \
    --config_file $STARVLA_ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 1 \
    $STARVLA_ROOT/starVLA/training/train_starvla.py \
    --config_yaml $CONFIG
" > "$LOG" 2>&1 &

TRAIN_PID=$!
echo "  Training launched: PID=$TRAIN_PID"
echo "  Log: $LOG"
echo "  Output: $OUTPUT_DIR"

# Wait a bit and check
sleep 10
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo -e "  ${GREEN}Training process is running.${NC}"
    echo ""
    echo "Monitor with:"
    echo "  tail -f $LOG"
    echo ""
    echo "Check status:"
    echo "  ps -p $TRAIN_PID -o pid,stat,%cpu,%mem,etime"
else
    echo -e "  ${RED}Training may have failed to start.${NC}"
    echo "  Last log lines:"
    tail -30 "$LOG"
fi

echo ""
echo "============================================"
echo "Setup complete. Run ID: $RUN_ID"
echo "============================================"
