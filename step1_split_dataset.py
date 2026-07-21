"""Step 1: Split gamepad dataset into train32 + val8."""
import paramiko, subprocess, sys

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.130', username='jjk', password='ym123456', timeout=10, look_for_keys=False, allow_agent=False)

i, o, e = c.exec_command(
    'source /home/jjk/miniconda3/etc/profile.d/conda.sh && '
    'cd /home/jjk/桌面/Project/lerobot-main && '
    'conda run --no-capture-output -n nova2_lerobot lerobot-edit-dataset '
    '--repo_id jjk/nova2_gamepad_6dof_raw_20260717_092054 '
    '--root /home/jjk/dobot_nova2_project/datasets/nova2_gamepad_6dof_raw_20260717_092054 '
    '--operation.type split '
    "--operation.splits '{\"train\":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],\"val\":[32,33,34,35,36,37,38,39]}' "
    '--new_root /home/jjk/dobot_nova2_project/datasets/.gamepad40_split_workspace '
    '--push_to_hub false 2>&1',
    timeout=180)
out = o.read().decode('utf-8', errors='replace')
err = e.read().decode('utf-8', errors='replace')
print('STDOUT:', out[:2000])
if err.strip():
    print('STDERR:', err.strip()[:1000])

# Move to final locations
i2, o2, e2 = c.exec_command(
    'mv /home/jjk/dobot_nova2_project/datasets/.gamepad40_split_workspace/train '
    '/home/jjk/dobot_nova2_project/datasets/nova2_gamepad_train32_20260717 && '
    'mv /home/jjk/dobot_nova2_project/datasets/.gamepad40_split_workspace/val '
    '/home/jjk/dobot_nova2_project/datasets/nova2_gamepad_val8_20260717 && '
    'rmdir /home/jjk/dobot_nova2_project/datasets/.gamepad40_split_workspace 2>/dev/null; '
    'echo "MOVE_DONE"',
    timeout=10)
print(o2.read().decode())

c.close()
