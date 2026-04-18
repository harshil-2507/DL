#!/bin/bash
#SBATCH --job-name=d4         # 🔁 CHANGE
#SBATCH --partition=gpu
#SBATCH --output=/home/rahuldixit/aksh/dl_project/logs/o4_code4_%j.log  # 🔁 CHANGE
#SBATCH --error=/home/rahuldixit/aksh/dl_project/logs/e4_code4_%j.log    # 🔁 CHANGE
#SBATCH --gres=shard:10                      # 🔁 CHANGE ( no of charades if needed GPUs)
#SBATCH --cpus-per-task=16                   # 🔁 CHANGE plus per task. I number 
#SBATCH --mem=64G                            # 🔁 CHANGE if needed 
#SBATCH --time=12:00:00                      # 🔁 CHANGE (HH:MM:SS)

echo "=========================================="
echo " JOB STARTED ON: $(date)"
echo " Agent: WCE_code3_final (ResNet/ViT/Swin)"
echo "=========================================="

module purge
module load anaconda3-2024.2 || module load anaconda3

eval "$(conda shell.bash hook)"
source /nfsroot/apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh 2>/dev/null || \
  source $(conda info --base)/etc/profile.d/conda.sh
conda config --set auto_activate_base false

# ─────────────────────────────────────────────
# 🔹 Environment Setup
# ─────────────────────────────────────────────
ENV_NAME=wce_env2   # 🔁 CHANGE if needed

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Env '${ENV_NAME}' exists. Activating..."
else
    echo "Env '${ENV_NAME}' not found. Creating..."
    conda create -n ${ENV_NAME} python=3.10 -y
fi

conda activate ${ENV_NAME}

if [[ "$CONDA_DEFAULT_ENV" != "${ENV_NAME}" ]]; then
    echo "ERROR: Failed to activate '${ENV_NAME}'"; exit 1
fi
echo "Active env: $CONDA_DEFAULT_ENV"

# ─────────────────────────────────────────────
# KEY FIX: Compute nodes have no internet.
# Only install packages if torch is NOT already
# importable (i.e. env was freshly created).
# ─────────────────────────────────────────────
if python -c "import torch" 2>/dev/null; then
    echo "torch already installed — skipping pip installs."
else
    echo "torch not found — installing packages..."
    echo "NOTE: This requires internet. Run from login node if this fails."
    pip install --upgrade --no-cache-dir pip
    pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
    pip install --no-cache-dir \
        nvidia-nvjitlink-cu12 \
        nvidia-cuda-runtime-cu12 \
        nvidia-cublas-cu12 \
        nvidia-cudnn-cu12
    pip install --no-cache-dir \
        "numpy==1.26.4" scipy scikit-learn tqdm matplotlib seaborn pandas opencv-python
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

for d in $(find $HOME/.local/lib -type f -name "libnvJitLink*" 2>/dev/null | xargs -I{} dirname {} | sort -u); do
    export LD_LIBRARY_PATH="${d}:$LD_LIBRARY_PATH"
    echo "libnvJitLink found at: ${d}"
done

export MPLBACKEND=Agg

# ─────────────────────────────────────────────
# 🔹 GPU Check
# ─────────────────────────────────────────────
nvidia-smi
python -c "
import torch
print('PyTorch :', torch.__version__)
print('CUDA    :', torch.cuda.is_available())
print('GPU     :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
" || { echo "torch import failed — abort."; exit 1; }

# ─────────────────────────────────────────────
# 🔹 Run your script
# ─────────────────────────────────────────────
SCRIPT_PATH=/home/rahuldixit/aksh/dl_project/4code_eval.py

# Create output directories just in case
mkdir -p /home/rahuldixit/aksh/dl_project/logs
mkdir -p /home/rahuldixit/aksh/dl_project/outputs_code4
mkdir -p /home/rahuldixit/aksh/dl_project/saved_models_code4

echo "=========================================="
echo " STARTING WCE_code4_final"
echo "=========================================="
echo "Running script: $SCRIPT_PATH"
python $SCRIPT_PATH
nvidia-smi

echo "=========================================="
echo " JOB FINISHED ON: $(date)"
echo "=========================================="

