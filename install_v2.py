import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("106.12.176.186", username="root", password="123456fF", timeout=30)

def run(cmd, timeout_val=600):
    print(f"\n>>> {cmd[:100]}")
    stdin, stdout, stderr = c.exec_command(cmd, get_pty=True, timeout=timeout_val)
    all_out = []
    for line in iter(stdout.readline, ""):
        s = line.strip()
        all_out.append(s)
        if len(s) > 0:
            print(f"    {s[:150]}")
    rc = stdout.channel.recv_exit_status()
    err = stderr.read().decode().strip()
    if err:
        print(f"    STDERR: {err[:200]}")
    return rc, all_out

# Kill all pip processes
run("pkill -9 -f pip 2>/dev/null; sleep 1; echo 'cleaned'")

# Install PyTorch with CUDA 12.1 (compatible with 535 driver)
run("pip3 uninstall torch torchvision -y 2>/dev/null; pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121", timeout_val=600)

# Install other packages
run("pip3 install numpy scipy scikit-learn tqdm matplotlib -q")

# Verify CUDA
print("\n=== Verify GPU ===")
rc, out = run("python3 -c 'import torch; print(\"torch\", torch.__version__); print(\"CUDA\", torch.cuda.is_available())'")
if rc == 0:
    run("python3 -c 'import torch; print(\"GPU:\", torch.cuda.get_device_name(0))'")

# Verify all packages
run("python3 -c 'import numpy,scipy,sklearn,tqdm,matplotlib; print(\"all packages OK\")'")

# Test training
print("\n=== Test training ===")
run("cd /root/cwru-working-paper/src && python3 -c 'from train import run_experiment; r=run_experiment(\"2d\",\"recording\",\"none\",42); print(\"acc=\"+str(r[\"accuracy\"])+\" f1=\"+str(r[\"macro_f1\"]))'", timeout_val=600)

c.close()
