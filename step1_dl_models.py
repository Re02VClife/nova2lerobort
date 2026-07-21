"""Step 1: Download SmolVLA models to server."""
import paramiko

HOST = '192.168.0.130'
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username='jjk', password='ym123456', timeout=10, look_for_keys=False, allow_agent=False)
print('Connected')

dl_script = r'''#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
source /home/jjk/miniconda3/etc/profile.d/conda.sh
MODEL_DIR="$HOME/桌面/Project/lerobot-main/models"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

echo "=== Download SmolVLM2-500M-Video-Instruct ==="
if [ -f "$MODEL_DIR/SmolVLM2-500M-Video-Instruct/config.json" ]; then
    echo "Already exists, skipping."
else
    conda run -n nova2_lerobot hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --local-dir "$MODEL_DIR/SmolVLM2-500M-Video-Instruct"
    echo "Done: SmolVLM2"
fi

echo ""
echo "=== Download smolvla_base ==="
if [ -f "$MODEL_DIR/smolvla_base/config.json" ]; then
    echo "Already exists, skipping."
else
    conda run -n nova2_lerobot hf download lerobot/smolvla_base --local-dir "$MODEL_DIR/smolvla_base"
    echo "Done: smolvla_base"
fi

echo ""
echo "Models:"
ls -la "$MODEL_DIR/"
echo ""
echo "DOWNLOAD_COMPLETE"
'''

i, o, e = c.exec_command('cat > /tmp/dl_models.sh', timeout=10)
i.write(dl_script)
i.close()
o.read(); e.read()

i, o, e = c.exec_command('chmod +x /tmp/dl_models.sh && bash /tmp/dl_models.sh 2>&1', timeout=900)

import time
start = time.time()
while not o.channel.exit_status_ready():
    if o.channel.recv_ready():
        data = o.channel.recv(8192).decode('utf-8', errors='replace')
        print(data, end='', flush=True)
    if time.time() - start > 900:
        break
    import select
    time.sleep(0.2)

remaining = o.read().decode('utf-8', errors='replace')
if remaining:
    print(remaining)
err = e.read().decode('utf-8', errors='replace')
if err.strip():
    print('STDERR:', err[-500:])

c.close()
print('\nDone.')
