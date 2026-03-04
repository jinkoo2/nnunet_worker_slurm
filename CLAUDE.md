# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`nnunet_worker_slurm` — nnUNet training worker that integrates with `nnunet_dashboard` and runs preprocessing + training as **SLURM batch jobs** rather than direct subprocesses. The worker process itself is lightweight: it registers with the dashboard, polls for assigned jobs, submits sbatch scripts, monitors SLURM job status + progress log files, and reports progress back to the dashboard.

## Commands

```bash
# Setup
conda create -n nnunet_trainer python=3.12 -y
conda activate nnunet_trainer
pip install nnunetv2
pip install -r requirements.txt

# Run (SLURM commands must be in PATH)
module load slurm
cp .env.example .env
# Edit .env — set DASHBOARD_URL, DASHBOARD_API_KEY, WORKER_NAME, DATA_DIR, partitions
conda activate nnunet_trainer
python main.py
```

## Architecture

**Stack:** Python 3.12, requests, pydantic-settings. No FastAPI — pure worker process.

**Key files:**
- `main.py` — entry point, sets up logging, calls `worker.run()`
- `app/config.py` — Pydantic `BaseSettings` from `.env` (includes SLURM settings)
- `app/dashboard_client.py` — `requests` wrapper for all dashboard API calls
- `app/notifier.py` — Google Chat webhook notifications (fire-and-forget)
- `app/slurm.py` — SLURM operations: `sbatch`, `squeue_state`, `wait_for_slurm_job`, script generation
- `app/trainer.py` — dataset extraction, SLURM-based preprocessing/training, log parsing, model export
- `app/worker.py` — main loop: register → heartbeat thread → poll → execute

## SLURM Partitions (nvwulf cluster)

| Partition | Nodes | Time Limit | GPUs |
|---|---|---|---|
| `b40x4` | 9 | 8:00:00 | RTX Pro 6000 Blackwell ×4 per node |
| `b40x4-long` | 9 | 2-00:00:00 | RTX Pro 6000 Blackwell ×4 per node |
| `debug-b40x4` | 9 | 1:00:00 | RTX Pro 6000 Blackwell ×4 per node |
| `h200x4` | 4 | 8:00:00 | H200 ×4 per node |
| `h200x4-long` | 4 | 2-00:00:00 | H200 ×4 per node |
| `h200x8` | 2 | 8:00:00 | H200 ×8 per node |
| `h200x8-long` | 2 | 2-00:00:00 | H200 ×8 per node |

**Preprocessing:** uses `b40x4` partition — **no GPU requested** (CPU-only job). Configurable via `SLURM_PARTITION_PREPROCESS`.

**Training:** uses `b40x4-long` partition with 1 GPU. Configurable via `SLURM_PARTITION_TRAIN` / `SLURM_GPUS_TRAIN`.

## Worker Flow

```
startup → register with dashboard (retry up to 10x)
        → start heartbeat daemon thread (every HEARTBEAT_INTERVAL_S)

poll loop (every POLL_INTERVAL_S):
  GET /api/jobs?worker_id={id}&status=pending
  if job found → execute_job (blocking)

execute_job:
  1. PUT /api/jobs/{id}/status  → "assigned"
  2. GET /api/datasets/{id}     → get dataset_name
  3. GET /api/datasets/{id}/download → stream ZIP to disk
  4. Extract ZIP → DATA_DIR/raw/{dataset_name}/ + DATA_DIR/preprocessed/{dataset_name}/
  5. PUT status → "preprocessing"
     write DATA_DIR/slurm_scripts/{job_id}_preprocess.sh
     sbatch → SLURM job ID
     monitor thread: tail slurm_{job_id}.log → parse "Preprocessing case" → POST preprocessing_progress
     wait_for_slurm_job (poll squeue/sacct every 30s)
  6. PUT status → "training"
     for fold in 0..4:
       write DATA_DIR/slurm_scripts/{job_id}_train_{config}_fold{n}.sh
       sbatch → SLURM job ID
       monitor thread: tail training_log_*.txt → parse epochs → POST training_progress every epoch
       upload log text every 60s
       wait_for_slurm_job
       POST validation_result from fold_{n}/validation/summary.json
  7. PUT status → "uploading"
     nnUNetv2_export_model_to_zip (runs locally, not via SLURM)
     POST /api/jobs/{id}/model
  8. PUT status → "done"
  on exception → PUT status → "failed", error_message=str(e)
  on cancellation → scancel active SLURM job, do NOT overwrite DB status
```

## Data Layout

```
DATA_DIR/
├── downloads/{dataset_id}.zip          ← downloaded from dashboard
├── raw/Dataset###_Name/                ← raw images + dataset.json
├── preprocessed/Dataset###_Name/       ← nnUNetPlans.json + preprocessed data
│   └── preprocessing_completed.txt     ← flag: skip if present
├── results/Dataset###_Name/            ← trained models + logs (managed by nnUNet)
├── exports/{dataset_name}_{config}.zip ← model export zip before upload
├── logs/{job_id}/preprocess/
│   └── slurm_{slurm_job_id}.log        ← SLURM stdout (has "Preprocessing case" lines)
├── logs/{job_id}/fold_{n}/
│   └── slurm_{slurm_job_id}.log        ← SLURM stdout for training fold n
└── slurm_scripts/
    ├── {job_id}_preprocess.sh
    └── {job_id}_train_{config}_fold{n}.sh
```

## nnUNet Training Log Format

```
2026-03-01 22:05:50: Epoch 605
2026-03-01 22:05:50: Current learning rate: 0.00433
2026-03-01 22:08:05: train_loss -0.6752
2026-03-01 22:08:05: val_loss -0.4224
2026-03-01 22:08:05: Pseudo dice [0.8876, 0.8706]
2026-03-01 22:05:48: Epoch time: 125.29 s   ← marks end of epoch block
```

Log file location: `DATA_DIR/results/{dataset_name}/nnUNetTrainer__nnUNetPlans__{configuration}/fold_{fold}/training_log_*.txt`

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DASHBOARD_URL` | `http://localhost:9333` | Dashboard server URL |
| `DASHBOARD_API_KEY` | `changeme` | API key (X-Api-Key header) |
| `WORKER_NAME` | `worker-slurm01` | Unique display name in dashboard |
| `WORKER_HOSTNAME` | `""` | Optional hostname info |
| `GPU_NAME` | `""` | GPU model string (informational only) |
| `GPU_MEMORY_GB` | `0.0` | GPU VRAM in GB (informational only) |
| `CPU_CORES` | `0` | CPU core count (informational only) |
| `POLL_INTERVAL_S` | `30` | Seconds between job polls |
| `HEARTBEAT_INTERVAL_S` | `60` | Seconds between heartbeats |
| `DATA_DIR` | `/data/nnunet_trainer_data` | Working directory |
| `CONDA_ENV` | `nnunet_trainer` | Conda env with nnunetv2 |
| `NUM_PREPROCESSING_WORKERS` | `8` | `-np` argument for preprocessing |
| `SLURM_PARTITION_PREPROCESS` | `b40x4` | SLURM partition for preprocessing (CPU-only) |
| `SLURM_CPUS_PREPROCESS` | `16` | CPUs per preprocessing job |
| `SLURM_MEM_PREPROCESS` | `128G` | Memory for preprocessing job |
| `SLURM_TIME_PREPROCESS` | `8:00:00` | Time limit for preprocessing job |
| `SLURM_PARTITION_TRAIN` | `b40x4-long` | SLURM partition for training |
| `SLURM_CPUS_TRAIN` | `16` | CPUs per training job |
| `SLURM_MEM_TRAIN` | `128G` | Memory for training job |
| `SLURM_TIME_TRAIN` | `2-00:00:00` | Time limit per training fold job |
| `SLURM_GPUS_TRAIN` | `1` | Number of GPUs per training job |
| `SLURM_MAIL_USER` | `""` | Email for SLURM notifications (leave blank to disable) |
| `SLURM_MAIL_TYPE` | `ALL` | SLURM mail events (ALL, BEGIN, END, FAIL) |

## Differences from nnunet_worker_direct_gpu

| Aspect | direct_gpu | slurm |
|---|---|---|
| Preprocessing | `subprocess.Popen(preprocess.sh)` | `sbatch` → poll squeue |
| Training | `subprocess.Popen(train.sh)` | `sbatch` → poll squeue |
| Progress (preprocess) | parse subprocess stdout live | parse SLURM log file |
| Progress (training) | monitor `training_log_*.txt` | same (file polling) |
| Cancellation | `SIGTERM` to process group | `scancel <slurm_job_id>` |
| Model export | subprocess (local) | subprocess (local, same) |
| Scripts dir | `scripts/preprocess.sh`, `scripts/train.sh` | generated dynamically in `DATA_DIR/slurm_scripts/` |
| Conda profile | env var `CONDA_PROFILE` | auto-detected via `conda info --base` in scripts |

## Conventions

- SLURM scripts are generated fresh per-job-per-fold in `DATA_DIR/slurm_scripts/`
- `sbatch --parsable` used to capture numeric job ID
- `squeue` checked first; `sacct` used as fallback for terminal state after job leaves queue
- Monitor thread polls every 5s for training log updates; `wait_for_slurm_job` polls every 30s
- `JobCancelled` exception from `slurm.py` is caught in `worker.py` — same as direct_gpu
