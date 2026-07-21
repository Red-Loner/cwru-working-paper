import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("106.12.176.186", username="root", password="123456fF", timeout=30)

def run(cmd):
    print(f"$ {cmd}")
    stdin, stdout, stderr = c.exec_command(cmd, get_pty=True, timeout=120)
    lines = stdout.read().decode()
    print(lines[:3000])
    rc = stdout.channel.recv_exit_status()
    return rc

print("=== pip processes ===")
run("ps aux | grep pip | grep -v grep")

print("\n=== installed packages ===")
run("pip3 list 2>/dev/null | grep -E 'torch|numpy|scipy|sklearn|tqdm|matplotlib'")

print("\n=== GPU ===")
run("nvidia-smi --query-gpu=name,memory.total --format=csv")

# If torch is not installed, wait for it
run("python3 -c 'import torch; print(torch.__version__)' 2>&1 || echo 'torch not yet installed'")

c.close()
