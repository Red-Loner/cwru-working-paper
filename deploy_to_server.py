import paramiko
import os
import time

HOST = "106.12.176.186"
USER = "root"
PASSWORD = "123456fF"

LOCAL_REPO = r"E:\Git\cwru-working-paper"
LOCAL_DATA = r"D:\CWRU_Bearing_Data\CWRU_Bearing_Data\Source_Datasets"
REMOTE_ROOT = "/root/cwru-working-paper"

def ssh_connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    return c

def sftp_connect():
    t = paramiko.Transport((HOST, 22))
    t.connect(username=USER, password=PASSWORD)
    return paramiko.SFTPClient.from_transport(t)

def run_cmd(ssh, cmd, desc=""):
    print(f"\n[{desc}] $ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
    for line in stdout:
        print(f"  {line.strip()}")
    exit_code = stdout.channel.recv_exit_status()
    if exit_code != 0:
        err = stderr.read().decode()
        if err:
            print(f"  STDERR: {err.strip()}")
    return exit_code

def upload_dir(sftp, local_dir, remote_dir, desc=""):
    print(f"\nUploading {local_dir} -> {remote_dir} {desc}")
    run_cmd(ssh, f"mkdir -p {remote_dir}", "mkdir")
    for root, dirs, files in os.walk(local_dir):
        rel = os.path.relpath(root, local_dir)
        if rel == ".":
            remote_sub = remote_dir
        else:
            remote_sub = remote_dir + "/" + rel.replace("\\", "/")
        run_cmd(ssh, f"mkdir -p '{remote_sub}'")
        for f in files:
            if f.endswith(".pyc") or f == "__pycache__":
                continue
            local = os.path.join(root, f)
            remote = remote_sub + "/" + f
            try:
                sftp.put(local, remote)
            except Exception as e:
                print(f"  FAIL {local}: {e}")
    print(f"  Done: {local_dir}")

print("Connecting SSH...")
ssh = ssh_connect()
print("Connected.")

# Step 1: Upload code repo
print("\n=== Uploading code repo ===")
sftp = sftp_connect()
upload_dir(sftp, LOCAL_REPO, REMOTE_ROOT)
sftp.close()

# Step 2: Upload data
print("\n=== Uploading CWRU data ===")
sftp = sftp_connect()
upload_dir(sftp, LOCAL_DATA, REMOTE_ROOT + "/data_manifest/raw")
sftp.close()

# Step 3: Run setup script
print("\n=== Running setup ===")
run_cmd(ssh, "bash /root/cwru-working-paper/setup_server.sh", "setup_server.sh")

# Step 4: Verify
print("\n=== Verifying ===")
run_cmd(ssh, "python3 -c \"import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\"", "GPU check")
run_cmd(ssh, "python3 -c \"import sys; sys.path.insert(0,'/root/cwru-working-paper/src'); from preprocess import build_datasets; w,l,ids,recs=build_datasets(42); print(f'{len(w)} windows, {len(recs)} recordings, {len(set(l.tolist()))} classes')\"", "Data check")

# Step 5: Start training
print("\n=== Starting training ===")
run_cmd(ssh, "cd /root/cwru-working-paper/src && nohup python3 run_all.py > /tmp/pipeline.log 2>&1 &", "Start pipeline")
run_cmd(ssh, "sleep 3; cat /tmp/pipeline.log | tail -5", "Pipeline startup check")

print("\n=== Deployment complete! ===")
print("Check progress: ssh root@106.12.176.186 'cat /tmp/pipeline.log | tail -20'")

ssh.close()
