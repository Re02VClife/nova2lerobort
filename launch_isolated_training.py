"""Set up and launch isolated StarVLA training on gamepad dataset."""
import paramiko, time, sys

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.130', username='jjk', password='ym123456', timeout=10, look_for_keys=False, allow_agent=False)

CI = "source /home/jjk/miniconda3/etc/profile.d/conda.sh"

def run(cmd, timeout=30):
    i, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', errors='replace')
    err = e.read().decode('utf-8', errors='replace')
    if err.strip():
        print("  [stderr]", err.strip()[:300])
    return out

print("=" * 60)
print("STEP 1: Split gamepad dataset into train(32) + val(8)")
print("=" * 60)

# Build the split script
split_cmd = f"""#!/usr/bin/env bash
set -euo pipefail
LEROBOT_ROOT="/home/jjk/桌面/Project/lerobot-main"
ENV_NAME="nova2_lerobot"
DATA_ROOT="/home/jjk/dobot_nova2_project/datasets"
SRC="$DATA_ROOT/nova2_gamepad_6dof_raw_20260717_092054"
WORK="$DATA_ROOT/.gamepad40_split_workspace"
TRAIN_DST="$DATA_ROOT/nova2_gamepad_train32_20260717"
VAL_DST="$DATA_ROOT/nova2_gamepad_val8_20260717"
REPO_ID="jjk/nova2_gamepad_6dof_raw_20260717_092054"

TRAIN_EPS='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]'
VAL_EPS='[32,33,34,35,36,37,38,39]'
SPLITS='{{"train":{TRAIN_EPS},"val":{VAL_EPS}}}'

test -d "$SRC" || {{ echo "source_missing=$SRC"; exit 1; }}

for dst in "$TRAIN_DST" "$VAL_DST"; do
    if [[ -e "$dst" ]]; then
        mv "$dst" "${{dst}}.OLD_$(date +%Y%m%d_%H%M%S)"
    fi
done
rm -rf "$WORK"
mkdir -p "$WORK"

cd "$LEROBOT_ROOT"
echo "Splitting $SRC -> train(32ep) + val(8ep)"
{CI} && conda run --no-capture-output -n "$ENV_NAME" lerobot-edit-dataset --repo_id "$REPO_ID" --root "$SRC" --operation.type split --operation.splits "$SPLITS" --new_root "$WORK" --push_to_hub false

test -d "$WORK/train" || {{ echo "train_missing"; exit 1; }}
test -d "$WORK/val" || {{ echo "val_missing"; exit 1; }}

mv "$WORK/train" "$TRAIN_DST"
mv "$WORK/val" "$VAL_DST"
rmdir "$WORK" 2>/dev/null || true

echo "train_dataset=$TRAIN_DST"
echo "val_dataset=$VAL_DST"
echo "split=PASS"
"""

run(f"cat > /tmp/split_gamepad.sh << 'ENDSCRIPT'\n{split_cmd}\nENDSCRIPT")
result = run(f"bash /tmp/split_gamepad.sh 2>&1")
print(result)

# Verify split
print("\nVerifying split...")
run("cat > /tmp/verify_split.py << 'ENDPY'\n"
    "import pyarrow.parquet as pq\n"
    "from collections import Counter\n"
    "for name, path in [('train', '/home/jjk/dobot_nova2_project/datasets/nova2_gamepad_train32_20260717/data/chunk-000/file-000.parquet'), ('val', '/home/jjk/dobot_nova2_project/datasets/nova2_gamepad_val8_20260717/data/chunk-000/file-000.parquet')]:\n"
    "    t = pq.read_table(path)\n"
    "    epis = t.column('episode_index').to_pylist()\n"
    "    cnt = Counter(epis)\n"
    "    print(f'{name}: {t.num_rows} frames, {len(cnt)} episodes ({sorted(cnt.keys())[:3]}...{sorted(cnt.keys())[-3:]})')\n"
    "ENDPY")
result = run(f"{CI} && conda run --no-capture-output -n nova2_lerobot python3 /tmp/verify_split.py")
print(result)

print("\n" + "=" * 60)
print("STEP 2: Register datasets in StarVLA data registry")
print("=" * 60)

# Backup and update data_config.py
run("cp ~/starVLA/examples/Nova2/train_files/data_registry/data_config.py ~/starVLA/examples/Nova2/train_files/data_registry/data_config.py.before_gamepad40_$(date +%Y%m%d_%H%M%S).bak")
result = run("grep -c 'nova2_gamepad' ~/starVLA/examples/Nova2/train_files/data_registry/data_config.py 2>/dev/null")
if '0' in result.strip():
    # Add new dataset registrations
    new_entries = """
DATASET_NAMED_MIXTURES["nova2_gamepad_train32_20260717"] = [("nova2_gamepad_train32_20260717", 1.0, "dobot_nova2_delta_eef")]
DATASET_NAMED_MIXTURES["nova2_gamepad_val8_20260717"] = [("nova2_gamepad_val8_20260717", 1.0, "dobot_nova2_delta_eef")]
"""
    run("cat >> ~/starVLA/examples/Nova2/train_files/data_registry/data_config.py << 'ENDREG'\n" + new_entries + "\nENDREG")
    print("Registered nova2_gamepad_train32_20260717 and nova2_gamepad_val8_20260717")
else:
    print("Already registered, skipping")

print("\n" + "=" * 60)
print("STEP 3: Create training config")
print("=" * 60)

# Create isolated training config
run("cat > /tmp/starvla_nova2_gamepad_train32.yaml << 'ENDCFG'\n"
    "run_id: nova2_gamepad_train32_isolated\n"
    "run_root_dir: /home/jjk/starVLA/results/Checkpoints\n"
    "seed: 42\n"
    "trackers:\n"
    "- jsonl\n"
    "wandb_entity: jinhuiye\n"
    "wandb_project: StarVLA_Libero\n"
    "is_debug: false\n"
    "framework:\n"
    "  name: QwenGR00T\n"
    "  qwenvl:\n"
    "    base_vlm: /home/jjk/starVLA/playground/Pretrained_models/Qwen2.5-VL-3B-Instruct\n"
    "    attn_implementation: flash_attention_2\n"
    "    vl_hidden_dim: 2048\n"
    "  dino:\n"
    "    dino_backbone: dinov2_vits14\n"
    "  action_model:\n"
    "    action_model_type: DiT-B\n"
    "    action_hidden_dim: 1024\n"
    "    hidden_size: 1024\n"
    "    add_pos_embed: true\n"
    "    max_seq_len: 1024\n"
    "    action_dim: 7\n"
    "    state_dim: 13\n"
    "    future_action_window_size: 15\n"
    "    action_horizon: 16\n"
    "    past_action_window_size: 0\n"
    "    repeated_diffusion_steps: 1\n"
    "    noise_beta_alpha: 1.5\n"
    "    noise_beta_beta: 1.0\n"
    "    noise_s: 0.999\n"
    "    num_timestep_buckets: 1000\n"
    "    num_inference_timesteps: 4\n"
    "    num_target_vision_tokens: 32\n"
    "    diffusion_model_cfg:\n"
    "      cross_attention_dim: 2048\n"
    "      dropout: 0.2\n"
    "      final_dropout: true\n"
    "      interleave_self_attention: true\n"
    "      norm_type: ada_norm\n"
    "      num_layers: 16\n"
    "      output_dim: 1024\n"
    "      positional_embeddings: null\n"
    "    gripper_loss_weight: 5.0\n"
    "    gripper_channel_idx: 6\n"
    "  reduce_in_full_precision: true\n"
    "datasets:\n"
    "  vlm_data:\n"
    "    dataset_py: vlm_datasets\n"
    "    dataformat: llava_json\n"
    "    dataset_use: asv2_conversation_en,asv2_detailed_description_en,asv2_region_captioning_en,coco_internvl_longcap_en,coco_karpathy_train_567_en,coco_negative_gpt4o_en,coco_poetry_zh,coco_rem_en_zh,cocorem_exist_yorn_en,cocotextv2_en,cocotextv2_gpt4o_en,okvqa_en,refcoco_grounding_aug_en,refcoco_grounding_en,tallyqa_coco_en,toloka_grounding_aug_en,vqav2_en,vsr_en\n"
    "    eval_dataset: aokvqa_cauldron_llava_format\n"
    "    data_flatten: false\n"
    "    base_interval: 2\n"
    "    max_pixels: 12845056\n"
    "    min_pixels: 3136\n"
    "    model_max_length: 2048\n"
    "    model_type: qwen2.5vl\n"
    "    per_device_batch_size: 3\n"
    "  vla_data:\n"
    "    dataset_py: lerobot_datasets\n"
    "    data_root_dir: /home/jjk/dobot_nova2_project/datasets\n"
    "    data_mix: nova2_gamepad_train32_20260717\n"
    "    action_type: delta_ee\n"
    "    CoT_prompt: Your task is {instruction}. To identify the key objects for your task. Locate their bounding boxes in [x1,y1,x2,y2] format.\n"
    "    CoT_answer: bbox\n"
    "    default_image_resolution:\n"
    "    - 3\n"
    "    - 224\n"
    "    - 224\n"
    "    per_device_batch_size: 1\n"
    "    load_all_data_for_training: true\n"
    "    obs:\n"
    "    - image_0\n"
    "    include_state: true\n"
    "    obs_image_size:\n"
    "    - 224\n"
    "    - 224\n"
    "    action_mode: abs\n"
    "    lerobot_version: v3.0\n"
    "    delete_pause_frame: false\n"
    "    video_backend: torchvision_av\n"
    "    num_workers: 0\n"
    "trainer:\n"
    "  epochs: 100\n"
    "  max_train_steps: 100\n"
    "  num_warmup_steps: 10\n"
    "  save_interval: 100\n"
    "  eval_interval: 100\n"
    "  learning_rate:\n"
    "    base: 0.0001\n"
    "    action_model: 0.0001\n"
    "  lr_scheduler_type: cosine_with_min_lr\n"
    "  scheduler_specific_kwargs:\n"
    "    min_lr: 1.0e-06\n"
    "  freeze_modules: qwen_vl_interface\n"
    "  loss_scale:\n"
    "    vla: 1.0\n"
    "    vlm: 0.1\n"
    "  max_grad_norm: 1.0\n"
    "  warmup_ratio: 0.1\n"
    "  weight_decay: 0.0\n"
    "  logging_frequency: 1\n"
    "  gradient_clipping: 1.0\n"
    "  is_resume: false\n"
    "  optimizer:\n"
    "    betas:\n"
    "    - 0.9\n"
    "    - 0.95\n"
    "    eps: 1.0e-08\n"
    "    weight_decay: 1.0e-08\n"
    "  pretrained_checkpoint: /home/jjk/starVLA/playground/Pretrained_models/Qwen2.5-VL-GR00T-Nova2-init/checkpoints/steps_30000_nova2_init_model.safetensors\n"
    "  reload_modules: null\n"
    "  save_format: safetensors\n"
    "output_dir: /home/jjk/starVLA/results/Checkpoints/nova2_gamepad_train32_isolated\n"
    "ENDCFG")
print("Config written to /tmp/starvla_nova2_gamepad_train32.yaml")

print("\n" + "=" * 60)
print("STEP 4: Check GPU memory before launch")
print("=" * 60)
result = run("nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv,noheader")
print(result)

print("\n" + "=" * 60)
print("STEP 5: Launch training (smoke test: 100 steps)")
print("=" * 60)

# Launch training as background process
launch_cmd = f"""cd /home/jjk/starVLA && \
WANDB_MODE=disabled \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=4 \
nice -n 10 \
{CI} && conda run --no-capture-output -n starvla \
accelerate launch \
--config_file /home/jjk/starVLA/starVLA/config/deepseeds/deepspeed_zero2.yaml \
--num_processes 1 \
/home/jjk/starVLA/starVLA/training/train_starvla.py \
--config_yaml /tmp/starvla_nova2_gamepad_train32.yaml \
> /home/jjk/starVLA/results/nova2_gamepad_train32_isolated.log 2>&1 &"""

result = run(f"bash -c '{launch_cmd} echo LAUNCHED; sleep 3; cat /home/jjk/starVLA/results/nova2_gamepad_train32_isolated.log 2>/dev/null | tail -20'")
print(result)

print("\n" + "=" * 60)
print("STEP 6: Verify training started")
print("=" * 60)
time.sleep(5)
result = run("tail -30 /home/jjk/starVLA/results/nova2_gamepad_train32_isolated.log 2>/dev/null")
print(result)
result = run("ps aux | grep -E 'train_starvla|accelerate' | grep -v grep")
print("Training process:", result[:500] if result.strip() else "NOT FOUND - check log")

c.close()
print("\nDone. Check log: ~/starVLA/results/nova2_gamepad_train32_isolated.log")
