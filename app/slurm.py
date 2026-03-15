"""
SLURM job submission, monitoring, and script generation for nnUNet preprocessing and training.
"""
import logging
import subprocess
import textwrap
import threading
import time
from pathlib import Path
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)


class SlurmJobFailed(Exception):
    """Raised when a SLURM job ends in a non-COMPLETED terminal state."""


class SlurmJobTimeout(Exception):
    """Raised when a SLURM job ends with state TIMEOUT (can be resubmitted to continue)."""


class JobCancelled(Exception):
    """Raised when a cancel_event triggers scancel of a running SLURM job."""


# ---------------------------------------------------------------------------
# SLURM command wrappers
# ---------------------------------------------------------------------------

def sbatch(script_path: str) -> str:
    """Submit a SLURM script and return the numeric job ID string."""
    result = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    # --parsable output: "<jobid>" or "<jobid>;<cluster>" — take first field
    job_id = result.stdout.strip().split(";")[0]
    logger.info(f"Submitted SLURM job {job_id} ({script_path})")
    return job_id


def squeue_state(slurm_job_id: str) -> Optional[str]:
    """
    Return the current SLURM job state (e.g. RUNNING, PENDING, COMPLETED) or None.
    Falls back to sacct if the job is no longer in the squeue (recently finished).
    """
    result = subprocess.run(
        ["squeue", "--job", slurm_job_id, "--noheader", "--format=%T"],
        capture_output=True, text=True,
    )
    line = result.stdout.strip()
    if line:
        return line  # e.g. RUNNING, PENDING, COMPLETING, etc.

    # Not in queue — check sacct for terminal state
    result = subprocess.run(
        ["sacct", "--jobs", slurm_job_id, "--noheader",
         "--format=State", "--parsable2"],
        capture_output=True, text=True,
    )
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if lines:
        # sacct State may be "CANCELLED by 12345" — take the first word
        return lines[0].split()[0]
    return None


def scancel(slurm_job_id: str) -> None:
    """Cancel a SLURM job (fire-and-forget; errors are logged only)."""
    result = subprocess.run(["scancel", slurm_job_id], capture_output=True)
    if result.returncode != 0:
        logger.warning(f"scancel {slurm_job_id} returned exit {result.returncode}")
    else:
        logger.info(f"scancel {slurm_job_id} sent")


def wait_for_slurm_job(
    slurm_job_id: str,
    cancel_event: Optional[threading.Event],
    poll_interval: int = 30,
) -> None:
    """
    Block until the SLURM job reaches a terminal state.
    If cancel_event is set mid-wait, cancels the job via scancel and raises JobCancelled.
    Raises SlurmJobFailed for non-COMPLETED terminal states.
    """
    logger.info(f"Waiting for SLURM job {slurm_job_id} (polling every {poll_interval}s)...")
    while True:
        if cancel_event is not None and cancel_event.is_set():
            logger.info(f"Cancel requested — sending scancel {slurm_job_id}")
            scancel(slurm_job_id)
            raise JobCancelled(f"SLURM job {slurm_job_id} cancelled by user")

        state = squeue_state(slurm_job_id)
        logger.debug(f"SLURM job {slurm_job_id} state={state}")

        if state is None or state == "COMPLETED":
            logger.info(f"SLURM job {slurm_job_id} completed successfully")
            return
        if state == "TIMEOUT":
            raise SlurmJobTimeout(
                f"SLURM job {slurm_job_id} ended with state: TIMEOUT"
            )
        if state in ("FAILED", "CANCELLED", "OUT_OF_MEMORY",
                     "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"):
            raise SlurmJobFailed(
                f"SLURM job {slurm_job_id} ended with state: {state}"
            )
        # RUNNING, PENDING, COMPLETING, REQUEUED, etc. — keep waiting
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# SLURM script generation
# ---------------------------------------------------------------------------

def _slurm_header(
    job_name: str,
    output_log: str,
    partition: str,
    cpus: int,
    mem: str,
    time_limit: str,
    gpus: int = 0,
    mail_user: str = "",
    mail_type: str = "ALL",
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output_log}",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --nodes=1",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH -p {partition}",
        f"#SBATCH --mem={mem}",
    ]
    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    if mail_user:
        lines.append(f"#SBATCH --mail-type={mail_type}")
        lines.append(f"#SBATCH --mail-user={mail_user}")
    return "\n".join(lines)


def _conda_block(conda_env: str) -> str:
    return textwrap.dedent(f"""\
        # Load conda and activate environment
        module load miniconda
        source /lustre/nvwulf/software/miniconda3/etc/profile.d/conda.sh
        conda activate "{conda_env}"
        echo "Activated conda env: {conda_env}"
        echo "Python: $(which python)"
    """)


def write_preprocess_script(
    script_path: Path,
    dataset_num: str,
    log_dir: Path,
    data_dir: str,
    conda_env: str,
) -> None:
    """Write a SLURM batch script for nnUNetv2_preprocess (CPU-only, no GPU)."""
    # Use slurm_<jobid>.log — %j is replaced by SLURM at submission time
    slurm_log = log_dir / "slurm_%j.log"
    header = _slurm_header(
        job_name=f"nnunet_preprocess_{dataset_num}",
        output_log=str(slurm_log),
        partition=settings.SLURM_PARTITION_PREPROCESS,
        cpus=settings.SLURM_CPUS_PREPROCESS,
        mem=settings.SLURM_MEM_PREPROCESS,
        time_limit=settings.SLURM_TIME_PREPROCESS,
        gpus=0,  # preprocessing is CPU-only
        mail_user=settings.SLURM_MAIL_USER,
        mail_type=settings.SLURM_MAIL_TYPE,
    )
    body = textwrap.dedent(f"""\

        echo "===== nnUNet preprocessing SLURM job ====="
        echo "SLURM_JOB_ID:  $SLURM_JOB_ID"
        echo "Running on:    $(hostname)"
        echo "Dataset num:   {dataset_num}"
        echo ""

        export nnUNet_raw="{data_dir}/raw"
        export nnUNet_preprocessed="{data_dir}/preprocessed"
        export nnUNet_results="{data_dir}/results"
        export TORCH_COMPILE_DISABLE=1

        {_conda_block(conda_env)}

        mkdir -p "{log_dir}"

        echo ""
        echo "=== nnUNetv2_preprocess: Dataset {dataset_num} ==="
        echo ""

        ${{CONDA_PREFIX}}/bin/nnUNetv2_preprocess \\
            -d "{dataset_num}" \\
            -np "{settings.NUM_PREPROCESSING_WORKERS}" \\
            --verbose

        STATUS=$?
        echo ""
        echo "Exit code: $STATUS"
        exit $STATUS
    """)
    script_path.write_text(header + "\n" + body)
    script_path.chmod(0o755)
    logger.info(f"Preprocessing SLURM script written to {script_path}")


def write_train_script(
    script_path: Path,
    dataset_num: str,
    configuration: str,
    fold: int,
    log_dir: Path,
    data_dir: str,
    conda_env: str,
) -> None:
    """Write a SLURM batch script for nnUNetv2_train (single fold, with GPU)."""
    slurm_log = log_dir / "slurm_%j.log"
    header = _slurm_header(
        job_name=f"nnunet_tr_{dataset_num}_{configuration}_f{fold}",
        output_log=str(slurm_log),
        partition=settings.SLURM_PARTITION_TRAIN,
        cpus=settings.SLURM_CPUS_TRAIN,
        mem=settings.SLURM_MEM_TRAIN,
        time_limit=settings.SLURM_TIME_TRAIN,
        gpus=settings.SLURM_GPUS_TRAIN,
        mail_user=settings.SLURM_MAIL_USER,
        mail_type=settings.SLURM_MAIL_TYPE,
    )
    body = textwrap.dedent(f"""\

        echo "===== nnUNet training SLURM job ====="
        echo "SLURM_JOB_ID:    $SLURM_JOB_ID"
        echo "Running on:      $(hostname)"
        echo "GPU:             $CUDA_VISIBLE_DEVICES"
        echo "Dataset num:     {dataset_num}"
        echo "Configuration:   {configuration}"
        echo "Fold:            {fold}"
        echo ""

        export nnUNet_raw="{data_dir}/raw"
        export nnUNet_preprocessed="{data_dir}/preprocessed"
        export nnUNet_results="{data_dir}/results"
        export TORCH_COMPILE_DISABLE=1
        export TMPDIR="{data_dir}/tmp"
        mkdir -p "$TMPDIR"

        {_conda_block(conda_env)}

        echo "nnUNetv2_train: ${{CONDA_PREFIX}}/bin/nnUNetv2_train"
        echo ""
        echo "=== nnUNetv2_train: {dataset_num} {configuration} fold {fold} ==="
        echo ""

        ${{CONDA_PREFIX}}/bin/nnUNetv2_train \\
            -device cuda \\
            -num_gpus {settings.SLURM_GPUS_TRAIN} \\
            --c \\
            "{dataset_num}" \\
            "{configuration}" \\
            "{fold}"

        STATUS=$?
        echo ""
        echo "Exit code: $STATUS"
        exit $STATUS
    """)
    script_path.write_text(header + "\n" + body)
    script_path.chmod(0o755)
    logger.info(f"Training SLURM script written to {script_path}")
