"""Download SmolVLA models from HF mirror and launch training."""
import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.130', username='jjk', password='ym123456', timeout=10, look_for_keys=False, allow_agent=False)

CI = 'source /home/jjk/miniconda3/etc/profile.d/conda.sh'

def run(cmd, timeout=60):
    i, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', errors='replace')
    err = e.read().decode('utf-8', errors='replace')
    return out, err

# Write download script to server
download_script = '''#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
source /home/jjk/miniconda3/etc/profile.d/conda.sh

MODEL_DIR="$HOME/桌面/Project/lerobot-main/models"
mkdir -p "$MODEL_DIR"

echo "============================================"
echo "Step 1: Download SmolVLM2-500M-Video-Instruct"
echo "============================================"
if [ -d "$MODEL_DIR/SmolVLM2-500M-Video-Instruct" ]; then
    echo "Already exists, skipping."
else
    conda run -n nova2_lerobot hf download \
        HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
        --local-dir "$MODEL_DIR/SmolVLM2-500M-Video-Instruct"
    echo "SmolVLM2-500M-Video-Instruct downloaded."
fi

echo ""
echo "============================================"
echo "Step 2: Download smolvla_base"
echo "============================================"
if [ -d "$MODEL_DIR/smolvla_base" ]; then
    echo "Already exists, skipping."
else
    conda run -n nova2_lerobot hf download \
        lerobot/smolvla_base \
        --local-dir "$MODEL_DIR/smolvla_base"
    echo "smolvla_base downloaded."
fi

echo ""
echo "DONE. Models in $MODEL_DIR"
ls -la "$MODEL_DIR/"
'''

run("cat > /tmp/download_smolvla_models.sh << 'ENDSCRIPT'\n" + download_script + "\nENDSCRIPT")
print("Download script written to server.")

print()
print("=== Launching model download (this may take 5-15 min) ===")
out, err = run("bash /tmp/download_smolvla_models.sh 2>&1", timeout=900)
print(out[-3000:] if len(out) > 3000 else out)
if err:
    print("STDERR:", err[-1000:] if len(err) > 1000 else err)

c.close()
