#!/bin/bash
# Start nnunet_worker_slurm using .env.b40x4
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE=".env.h200x4"

# Parse CONDA_ENV from the env file
CONDA_ENV=$(grep -E '^CONDA_ENV=' "$SCRIPT_DIR/$ENV_FILE" | cut -d= -f2 | tr -d ' \r')
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: CONDA_ENV not found in $ENV_FILE"
    exit 1
fi
echo "Using conda env: $CONDA_ENV"

# Load SLURM
module load slurm
echo "SLURM loaded: $(sbatch --version)"

# Load miniconda and initialize conda for this shell
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create conda environment if it doesn't exist
if ! conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
    echo "Creating conda environment: $CONDA_ENV (python=3.12)"
    conda create -n "$CONDA_ENV" python=3.12 -y
fi

# Activate environment
conda activate "$CONDA_ENV"
echo "Activated: $(which python) ($(python --version))"

# Install / update packages
pip install -r "$SCRIPT_DIR/requirements.txt"
pip install nnunetv2

# Start the worker
echo ""
echo "=== Starting nnunet_worker_slurm with $ENV_FILE ==="
echo ""
cd "$SCRIPT_DIR"
ENV_FILE="$ENV_FILE" python main.py
