# nnunet_worker_slurm

nnUNet training worker that integrates with [nnunet_dashboard](https://github.com/jinkoo2/nnunet_dashboard) and runs preprocessing + training as **SLURM batch jobs**. The worker process itself is lightweight: it registers with the dashboard, polls for assigned jobs, submits sbatch scripts, monitors SLURM job status and log files, and reports progress back to the dashboard.

## Features

- Preprocessing and all 5 training folds submitted as SLURM batch jobs (not local subprocesses)
- All 5 fold training jobs submitted to the queue simultaneously — folds run in parallel if the node allows
- Per-fold progress (epoch metrics) and log files streamed back to the dashboard in real time
- Automatic skip of download/extraction/preprocessing if already completed
- Cancellation support: `scancel` is issued for all active SLURM jobs when a job is cancelled from the dashboard
- Google Chat webhook notifications (optional)

## Requirements

- Linux with SLURM (`module load slurm` must make `sbatch`, `squeue`, `sacct`, `scancel` available)
- Miniconda / Conda (`module load miniconda`)
- Access to a running [nnunet_dashboard](https://github.com/jinkoo2/nnunet_dashboard) instance

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/jinkoo2/nnunet_worker_slurm.git
cd nnunet_worker_slurm
```

**2. Create the data directory**

```bash
mkdir -p /path/to/nnunet-data
```

Set this path as `DATA_DIR` in your `.env` file (next step).

**3. Configure the worker**

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```ini
DASHBOARD_URL=https://your-dashboard-host
DASHBOARD_API_KEY=your-api-key
WORKER_NAME=worker-slurm01        # unique name shown in the dashboard
DATA_DIR=/path/to/nnunet-data     # must exist and have sufficient storage
CONDA_ENV=nnunet_trainer          # conda env name (created automatically if missing)
```

See the [Configuration](#configuration) section for all available options.

## Starting the Worker

The simplest way is to use the provided start scripts, which handle module loading, conda environment creation, and dependency installation automatically:

```bash
bash start_worker_b40x4.sh        # uses .env.b40x4      (b40x4 partition)
bash start_worker_b40x4-long.sh   # uses .env.b40x4-long (b40x4-long partition)
```

Each script:
1. Reads `CONDA_ENV` from its `.env` file
2. Loads SLURM and Miniconda modules
3. Creates the conda environment if it does not exist
4. Installs/updates `requirements.txt` and `nnunetv2`
5. Starts `python main.py` with the selected `.env` file

**Manual start** (if you manage the environment yourself):

```bash
module load slurm
conda activate nnunet_trainer
python main.py
```

To use a specific `.env` file:

```bash
module load slurm
conda activate nnunet_trainer
ENV_FILE=.env.b40x4 python main.py
```

## Configuration

Copy `.env.example` to `.env` and edit as needed.

| Variable | Default | Description |
|---|---|---|
| `DASHBOARD_URL` | `http://localhost:9333` | Dashboard server URL |
| `DASHBOARD_API_KEY` | `changeme` | API key (`X-Api-Key` header) |
| `WORKER_NAME` | `worker-slurm01` | Unique display name in dashboard |
| `WORKER_HOSTNAME` | `""` | Optional hostname info |
| `GPU_NAME` | `""` | GPU model string (informational) |
| `GPU_MEMORY_GB` | `0.0` | GPU VRAM in GB (informational) |
| `CPU_CORES` | `0` | CPU core count (informational) |
| `POLL_INTERVAL_S` | `30` | Seconds between job polls |
| `HEARTBEAT_INTERVAL_S` | `60` | Seconds between heartbeats |
| `DATA_DIR` | `/data/nnunet_trainer_data` | Working directory for data/logs/scripts |
| `CONDA_ENV` | `nnunet_trainer` | Conda environment with nnunetv2 |
| `NUM_PREPROCESSING_WORKERS` | `8` | `-np` argument for preprocessing |
| `SLURM_PARTITION_PREPROCESS` | `b40x4` | SLURM partition for preprocessing (CPU-only) |
| `SLURM_CPUS_PREPROCESS` | `16` | CPUs for preprocessing job |
| `SLURM_MEM_PREPROCESS` | `128G` | Memory for preprocessing job |
| `SLURM_TIME_PREPROCESS` | `8:00:00` | Time limit for preprocessing job |
| `SLURM_PARTITION_TRAIN` | `b40x4-long` | SLURM partition for training |
| `SLURM_CPUS_TRAIN` | `16` | CPUs per training fold job |
| `SLURM_MEM_TRAIN` | `128G` | Memory per training fold job |
| `SLURM_TIME_TRAIN` | `2-00:00:00` | Time limit per training fold job |
| `SLURM_GPUS_TRAIN` | `1` | GPUs per training fold job |
| `SLURM_MAIL_USER` | `""` | Email for SLURM notifications (blank = disabled) |
| `SLURM_MAIL_TYPE` | `ALL` | SLURM mail events (`ALL`, `BEGIN`, `END`, `FAIL`) |
| `GOOGLE_CHAT_WEBHOOK_URL` | `""` | Google Chat webhook (blank = disabled) |

## SLURM Partitions (nvwulf cluster)

| Partition | Time Limit | GPUs |
|---|---|---|
| `b40x4` | 8:00:00 | RTX Pro 6000 Blackwell ×4 per node |
| `b40x4-long` | 2-00:00:00 | RTX Pro 6000 Blackwell ×4 per node |
| `debug-b40x4` | 1:00:00 | RTX Pro 6000 Blackwell ×4 per node |
| `h200x4` | 8:00:00 | H200 ×4 per node |
| `h200x4-long` | 2-00:00:00 | H200 ×4 per node |
| `h200x8` | 8:00:00 | H200 ×8 per node |
| `h200x8-long` | 2-00:00:00 | H200 ×8 per node |

Preprocessing uses `b40x4` (CPU-only, no GPU requested). Training uses `b40x4-long` with 1 GPU per fold.

## Worker Flow

```
startup → register with dashboard (retry up to 10x)
        → start heartbeat daemon thread

poll loop (every POLL_INTERVAL_S):
  GET /api/jobs?worker_id={id}&status=pending
  if job found → execute_job (blocking)

execute_job:
  1. PUT /api/jobs/{id}/status  → "assigned"
  2. GET /api/datasets/{id}     → get dataset_name
  3. GET /api/datasets/{id}/download → stream ZIP to disk (skip if cached)
  4. Extract ZIP → DATA_DIR/raw/{dataset_name}/ + DATA_DIR/preprocessed/{dataset_name}/
  5. PUT status → "preprocessing"
     sbatch preprocess script → monitor SLURM log → POST preprocessing_progress
     wait_for_slurm_job
  6. PUT status → "training"
     submit all 5 fold jobs simultaneously via sbatch
     monitor all folds in parallel (one thread per fold):
       parse training_log_*.txt → POST training_progress per epoch
       upload log text every 60s
     wait for all SLURM jobs concurrently
     POST validation_result for each fold
  7. PUT status → "uploading"
     nnUNetv2_export_model_to_zip (runs locally)
     POST /api/jobs/{id}/model
  8. PUT status → "done"
```

## Data Layout

```
DATA_DIR/
├── downloads/{dataset_id}.zip
├── raw/Dataset###_Name/
├── preprocessed/Dataset###_Name/
│   └── preprocessing_completed.txt   ← skip flag
├── results/Dataset###_Name/          ← nnUNet outputs
├── exports/{dataset_name}_{config}.zip
├── logs/{job_id}/preprocess/
│   └── slurm_{slurm_job_id}.log
├── logs/{job_id}/fold_{n}/
│   └── slurm_{slurm_job_id}.log
└── slurm_scripts/
    ├── {job_id}_preprocess.sh
    └── {job_id}_train_{config}_fold{n}.sh
```

## Architecture

| File | Role |
|---|---|
| `main.py` | Entry point |
| `app/config.py` | Pydantic `BaseSettings` from `.env` |
| `app/worker.py` | Main loop: register → heartbeat → poll → execute |
| `app/trainer.py` | Dataset setup, SLURM job submission, log parsing, model export |
| `app/slurm.py` | `sbatch`, `squeue_state`, `wait_for_slurm_job`, script generation |
| `app/dashboard_client.py` | `requests` wrapper for all dashboard API calls |
| `app/notifier.py` | Google Chat webhook notifications |
