"""Fix: Remove corrupted steps_data_index.pkl and verify datasets."""
import paramiko, base64

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.130', username='jjk', password='ym123456', timeout=10, look_for_keys=False, allow_agent=False)

def run(cmd, timeout=120):
    i, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', errors='replace')
    err = e.read().decode('utf-8', errors='replace')
    if err.strip():
        print("  ERR:", err.strip()[:500])
    return out

# 1. Remove cached pickle files
for ds in ['nova2_gamepad_train32_20260717', 'nova2_gamepad_val8_20260717']:
    cache = f'/home/jjk/dobot_nova2_project/datasets/{ds}/meta/steps_data_index.pkl'
    out = run(f'rm -f {cache} && echo deleted_cache_{ds} || echo failed_{ds}')
    print(out.strip())

# 2. Upload verification script via base64
verify_py = """import sys
sys.path.insert(0, '/home/jjk/starVLA')
sys.path.insert(0, '/home/jjk/starVLA/examples/Nova2/train_files/data_registry')
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from data_config import DobotNova2DeltaEefDataConfig
cfg = DobotNova2DeltaEefDataConfig()
fc = {'lerobot_version': 'v3.0', 'video_backend': 'torchvision_av', 'action_mode': 'abs'}

print('=== TRAIN ===')
ds = LeRobotSingleDataset(
    dataset_path='/home/jjk/dobot_nova2_project/datasets/nova2_gamepad_train32_20260717',
    modality_configs=cfg.modality_config(),
    transforms=cfg.transform(),
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    video_backend='torchvision_av',
    data_cfg=fc
)
print(f'len={len(ds)}')

print('=== VAL ===')
ds2 = LeRobotSingleDataset(
    dataset_path='/home/jjk/dobot_nova2_project/datasets/nova2_gamepad_val8_20260717',
    modality_configs=cfg.modality_config(),
    transforms=cfg.transform(),
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    video_backend='torchvision_av',
    data_cfg=fc
)
print(f'len={len(ds2)}')
"""

encoded = base64.b64encode(verify_py.encode()).decode()
run(f'echo {encoded} | base64 -d > /tmp/verify_fix.py')
print('Script uploaded.')

# 3. Run verification
cmd = 'source /home/jjk/miniconda3/etc/profile.d/conda.sh && conda run --no-capture-output -n starvla python3 /tmp/verify_fix.py'
print(run(cmd))

c.close()
print('Done.')
