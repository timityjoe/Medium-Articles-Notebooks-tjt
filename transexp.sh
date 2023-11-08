#!/bin/bash

# See https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html

# conda create --name conda39-transexp python=3.9
# conda activate conda39-transexp
# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113^C
# pip install -r requirements.txt

echo "Setting up ViT Transformer Explainability Environment..."
source activate base	
conda deactivate
conda activate conda39-transexp
echo "$PYTHON_PATH"
