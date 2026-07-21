import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("106.12.176.186", username="root", password="123456fF", timeout=30)

def run(cmd, timeout_val=300):
    print(f"\n$ {cmd[:80]}...")
    stdin, stdout, stderr = c.exec_command(cmd, get_pty=True, timeout=timeout_val)
    last = ""
    for line in iter(stdout.readline, ""):
        if line.strip():
            last = line.strip()
            if "%" in last or "Download" in last or "Installing" in last or "Success" in last or "ERROR" in last:
                print(f"  {last[:120]}")
    rc = stdout.channel.recv_exit_status()
    return rc

# Kill old torch install
run("kill -9 49589 49590 2>/dev/null; echo 'killed'")

# Clean install one by one
for pkg in ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "tqdm", "matplotlib"]:
    rc = run(f"pip3 install {pkg} -q 2>&1")
    if rc != 0:
        print(f"  FAILED: {pkg}")

# Verify
print("\n=== Verification ===")
run("python3 -c 'import torch; print(\"torch\",torch.__version__); print(\"CUDA\",torch.cuda.is_available(),torch.cuda.get_device_name(0))'")
run("python3 -c 'import numpy,scipy,sklearn,tqdm,matplotlib; print(\"all packages OK\")'")

print("\n=== Create server config ===")
cfg = '''DATA_ROOT = "/root/cwru-working-paper/data_manifest/raw"
WINDOW_LENGTH = 1024
SAMPLE_RATE = 12000
OVERLAP_RANDOM = 0.5
OVERLAP_RECORDING = 0.0
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_SEEDS = [42, 123, 456]
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = "cuda"
RESULTS_DIR = "../results"
'''
stdin, stdout, stderr = c.exec_command(f"cat > /root/cwru-working-paper/src/config.py << 'EOF'\n{cfg}\nEOF")
rc = stdout.channel.recv_exit_status()
print(f"Config: {'OK' if rc==0 else 'FAIL'}")

# Test data loading
print("\n=== Test data loading ===")
run("cd /root/cwru-working-paper/src && python3 -c 'from preprocess import build_datasets; w,l,ids,recs=build_datasets(42); print(f\"{len(w)} windows from {len(recs)} recordings\")'")

c.close()
