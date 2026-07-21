"""Check SmolVLA training environment on Linux server."""
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.130', username='jjk', password='ym123456', timeout=10, look_for_keys=False, allow_agent=False)

def run(cmd, timeout=20):
    i, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', errors='replace')
    err = e.read().decode('utf-8', errors='replace')
    return out.strip(), err.strip()

print('=== 1. Check LeRobot env ===')
out, err = run("source /home/jjk/miniconda3/etc/profile.d/conda.sh && conda run -n nova2_lerobot python -c \"import lerobot; print('lerobot installed OK')\" 2>&1")
print(out[:500])
if err:
    print('STDERR:', err[:500])

print()
print('=== 2. Check SmolVLA model dirs ===')
for p in [
    '/home/jjk/桌面/Project/lerobot-main/models',
    '/home/jjk/桌面/Project/lerobot-main/models/smolvla_base_nova2',
    '/home/jjk/桌面/Project/lerobot-main/models/smolvla_base_nova2_migrated',
    '/home/jjk/桌面/Project/lerobot-main/models/SmolVLM2-500M-Video-Instruct',
]:
    out, _ = run(f'test -d {p} && echo "EXISTS: {p}" || echo "NOT_FOUND: {p}"')
    print(out.strip())

print()
print('=== 3. Find any smolvla models anywhere on server ===')
out, _ = run('find /home/jjk -maxdepth 5 -type d -iname "*smolvla*" 2>/dev/null')
print(out[:2000] if out.strip() else 'None found')

print()
print('=== 4. Check lerobot train with smolvla policy ===')
out, err = run("source /home/jjk/miniconda3/etc/profile.d/conda.sh && conda run -n nova2_lerobot python -m lerobot.scripts.lerobot_train --help 2>&1 | head -40")
print(out[:2000])
if err:
    print('STDERR:', err[:500])

print()
print('=== 5. Check if policy.type=smolvla is supported ===')
out, err = run("source /home/jjk/miniconda3/etc/profile.d/conda.sh && conda run -n nova2_lerobot python -c \"from lerobot.policies.pretrained import PreTrainedPolicy; print([n for n in dir(PreTrainedPolicy) if 'smol' in n.lower()])\" 2>&1")
print(out[:500])
if err:
    print('STDERR:', err[:500])

print()
print('=== 6. Find SmolVLA policy source file ===')
out, _ = run('find /home/jjk/桌面/Project/lerobot-main -name "*.py" -path "*smol*" 2>/dev/null')
print(out[:1000] if out.strip() else 'None found (checking elsewhere...)')
out, _ = run('find /home/jjk/桌面/Project/lerobot-main/src/lerobot -name "*smol*" -o -name "*vla*" 2>/dev/null')
print(out[:1000] if out.strip() else 'None found')

print()
print('=== 7. Check StarVLA vs SmolVLA — what models exist? ===')
out, _ = run('ls -la /home/jjk/starVLA/playground/Pretrained_models/ 2>/dev/null')
print(out[:2000])
out, _ = run('ls -la /home/jjk/桌面/Project/lerobot-main/models/ 2>/dev/null')
print(out[:2000])

c.close()
