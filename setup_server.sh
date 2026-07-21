#!/bin/bash
set -e

echo "=== 1. Update system ==="
apt update -y

echo "=== 2. Install NVIDIA driver + CUDA ==="
apt install -y nvidia-driver-535 nvidia-utils-535

echo "=== 3. Install Python packages ==="
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install numpy scipy scikit-learn tqdm matplotlib

echo "=== 4. Verify GPU ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "=== 5. Create config ==="
cat > /root/cwru-working-paper/src/config.py << 'EOF'
DATA_ROOT = "/root/cwru-working-paper/data/raw"
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
EOF

echo "=== 6. Move data to correct path ==="
mkdir -p /root/cwru-working-paper/data/raw
if [ -d "/root/cwru-working-paper/data_manifest/raw/Source_Datasets" ]; then
    mv /root/cwru-working-paper/data_manifest/raw/Source_Datasets/* /root/cwru-working-paper/data/raw/
    rm -rf /root/cwru-working-paper/data_manifest/raw
elif [ -d "/root/cwru-working-paper/data_manifest/raw/01_Normal_Baseline" ]; then
    mv /root/cwru-working-paper/data_manifest/raw/* /root/cwru-working-paper/data/raw/
fi

echo "=== 7. Verify data ==="
python3 -c "
import os, glob
d = '/root/cwru-working-paper/data/raw'
files = sum(1 for _ in glob.glob(d + '/**/*.mat', recursive=True))
subdirs = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
print(f'Data path: {d}')
print(f'Subdirs: {subdirs}')
print(f'Total .mat files: {files}')
"

echo "=== Setup complete! ==="
echo "Run: cd /root/cwru-working-paper/src && python3 run_all.py"
