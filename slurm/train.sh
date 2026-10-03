#!/bin/bash
#SBATCH --job-name=epi-train
#SBATCH --partition=gpu-4-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# Uso: sbatch slurm/train.sh [config.yaml] [chave=valor ...]
# Espera models/ e o peso da checagem de AMP já baixados, porque o nó de GPU pode não ter internet.
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate epi
# O OpenCV precisa da libGL para carregar, e os nós de cálculo não a têm; ela vem do ambiente conda.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export YOLO_CONFIG_DIR="$HOME/.ultralytics"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

echo "nó: $(hostname) | commit: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python scripts/train.py "${1:-configs/train_yolov8s_640.yaml}" "${@:2}"
