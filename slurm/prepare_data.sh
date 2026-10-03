#!/bin/bash
#SBATCH --job-name=epi-prepare
#SBATCH --partition=intel-128
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Extrai, converte e divide o SH17 num nó de CPU. Espera data/raw/sh17.zip já baixado,
# porque os nós de cálculo podem não ter acesso à internet.
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate epi
# O OpenCV precisa da libGL para carregar, e os nós de cálculo não a têm; ela vem do ambiente conda.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export YOLO_CONFIG_DIR="$HOME/.ultralytics"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

echo "nó: $(hostname) | núcleos: $SLURM_CPUS_PER_TASK | commit: $(git rev-parse --short HEAD)"
python -m zipfile -e data/raw/sh17.zip data/raw/sh17
python scripts/prepare_sh17.py --workers "$SLURM_CPUS_PER_TASK"
python scripts/split_dataset.py
python scripts/dataset_fingerprint.py
git status --short
