#!/bin/bash
# Remote build and run script for ds4-cuda on DGX Spark
set -e

echo "=== Remote Build & Run ==="

MODEL_DIR="/home/xexnaor/.cache/huggingface/hub/models--nvidia--DeepSeek-V4-Flash-NVFP4/snapshots/e3cd60e7de98e9867116860d522499a728de1cf9"

# SSH into remote and execute commands
ssh xexnaor@10.0.0.2 << REMOTE_SCRIPT
set -e
cd ~/ds4-cuda-remote

# Clean and rebuild
rm -f *.o ds4 ds4-server ds4-bench ds4-eval ds4-agent
git reset --hard HEAD 2>/dev/null
git pull 2>/dev/null

make NVCC=/usr/local/cuda/bin/nvcc clean 2>&1
make NVCC=/usr/local/cuda/bin/nvcc ds4 2>&1

echo "=== Build complete ==="

echo "Model: ${MODEL_DIR}"

# Run model inspection
./ds4 --inspect --model "${MODEL_DIR}" --cuda 2>&1
echo "=== Inspection complete ==="

# Run inference test
./ds4 --model "${MODEL_DIR}" --cuda --prompt 'Hello' -n 1 2>&1
echo "=== Inference test complete ==="
REMOTE_SCRIPT
