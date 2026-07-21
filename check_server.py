import paramiko

HOST = "106.12.176.186"
USER = "root"
PASSWORD = "123456fF"

def ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    return c

def run(cmd, ssh_client=None):
    if ssh_client is None:
        ssh_client = ssh()
    print(f"$ {cmd}")
    stdin, stdout, stderr = ssh_client.exec_command(cmd, get_pty=True)
    for line in stdout:
        print(f"  {line.strip()}")
    exit_code = stdout.channel.recv_exit_status()
    if exit_code != 0:
        err = stderr.read().decode()
        if err.strip():
            print(f"  ERR: {err.strip()}")
    return exit_code

c = ssh()

print("=== 1. Check NVIDIA driver ===")
run("nvidia-smi 2>&1 | head -5", c)

print("\n=== 2. Check Python packages ===")
run("pip3 list 2>/dev/null | grep -E 'torch|numpy|scipy|sklearn|tqdm|matplotlib' | head -10", c)

print("\n=== 3. Check data ===")
run("ls -la /root/cwru-working-paper/data_manifest/raw/ 2>&1 | head -15", c)

print("\n=== 4. Check code ===")
run("ls /root/cwru-working-paper/src/ 2>&1", c)

print("\n=== 5. Check setup log ===")
run("tail -30 /tmp/setup.log 2>/dev/null || echo 'no setup log yet'", c)

c.close()
