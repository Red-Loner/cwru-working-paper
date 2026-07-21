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
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        err = stderr.read().decode()
        if err.strip():
            print(f"  ERR: {err.strip()}")
    return rc

c = ssh()

print("=== 1. Install Python packages ===")
run("pip3 install --upgrade pip -q", c)
run("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -5", c)
run("pip3 install numpy scipy scikit-learn tqdm matplotlib 2>&1 | tail -5", c)

print("\n=== 2. Create server config ===")
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
cmd = f"cat > /root/cwru-working-paper/src/config.py << 'PYEOF'\n{cfg}PYEOF"
stdin, stdout, stderr = c.exec_command(cmd)
stdout.channel.recv_exit_status()
print("  Config written.")

print("\n=== 3. Verify GPU + data ===")
run("python3 -c \"import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))\"", c)
run("python3 -c \"import sys; sys.path.insert(0,'/root/cwru-working-paper/src'); from preprocess import build_datasets; w,l,ids,recs=build_datasets(42); print(f'{len(w)} windows, {len(recs)} recordings, {len(set(l.tolist()))} classes')\"", c)

print("\n=== 4. Quick training test ===")
run("cd /root/cwru-working-paper/src && python3 -c \"from train import run_experiment; r=run_experiment('2d','recording','none',42); print(f'Test: acc={r[chr(39)+chr(97)+chr(99)+chr(99)+chr(117)+chr(114)+chr(97)+chr(99)+chr(121)+chr(39)]:.4f} f1={r[chr(39)+chr(109)+chr(97)+chr(99)+chr(114)+chr(111)+chr(95)+chr(102)+chr(49)+chr(39)]:.4f}')\"", c)

print("\n=== 5. Start full pipeline ===")
run("cd /root/cwru-working-paper/src && nohup python3 run_all.py > /tmp/pipeline.log 2>&1 & echo 'PID='$!", c)
run("sleep 10; head -30 /tmp/pipeline.log", c)

print("\n=== Done! ===")
print("Monitor: python3 -c \"import paramiko; ...\" or manual SSH")
c.close()
