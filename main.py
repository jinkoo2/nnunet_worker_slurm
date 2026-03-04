#!/usr/bin/env python3
"""
nnunet_worker_slurm — nnUNet training worker that submits preprocessing and
training jobs via SLURM, then monitors progress and reports back to the dashboard.

Usage:
    module load slurm
    conda activate nnunet_trainer
    python main.py
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from app.worker import run  # noqa: E402

if __name__ == "__main__":
    run()
