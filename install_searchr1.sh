#!/bin/bash
set -e 
conda create -y -n searchr1 python=3.9
source /opt/conda/etc/profile.d/conda.sh
conda activate searchr1

# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
# pip3 install flash-attn --no-build-isolation
# below command works somtimes but not others 
# pip3 install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl --force-reinstall --no-deps


pip install wandb
