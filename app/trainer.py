"""
Core training logic: dataset setup, SLURM job submission, log parsing.

Unlike nnunet_worker_direct_gpu, preprocessing and training are submitted as
SLURM batch jobs rather than run as direct subprocesses. Progress is monitored
by reading log files written by the running SLURM jobs.
"""
import json
import logging
import os
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

from .config import settings
from . import slurm
from .slurm import JobCancelled, SlurmJobFailed  # re-export for worker.py

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns for nnUNet training log
# ---------------------------------------------------------------------------
_RE_EPOCH = re.compile(r'Epoch (\d+)')
_RE_LR = re.compile(r'Current learning rate:\s*([0-9.e+\-]+)')
_RE_TRAIN_LOSS = re.compile(r'train_loss\s+([0-9.e+\-]+)')
_RE_VAL_LOSS = re.compile(r'val_loss\s+([0-9.e+\-]+)')
_RE_DICE = re.compile(r'Pseudo dice \[([^\]]+)\]')
_RE_EPOCH_TIME = re.compile(r'Epoch time:\s*([0-9.]+)\s*s')
_RE_PARENS = re.compile(r'\(([^)]+)\)')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_dataset_num(dataset_name: str) -> str:
    """Extract numeric ID string from 'Dataset###_Name' (e.g. '289' from 'Dataset289_Brain')."""
    m = re.search(r'Dataset(\d+)_', dataset_name)
    if not m:
        raise ValueError(f"Cannot extract dataset number from: {dataset_name!r}")
    return str(int(m.group(1)))  # strip leading zeros; nnUNet accepts plain int string


def get_nnunet_env() -> dict:
    """Build environment variables for nnUNet paths."""
    data_dir = settings.DATA_DIR
    env = os.environ.copy()
    env["nnUNet_raw"] = str(Path(data_dir) / "raw")
    env["nnUNet_preprocessed"] = str(Path(data_dir) / "preprocessed")
    env["nnUNet_results"] = str(Path(data_dir) / "results")
    env["TORCH_COMPILE_DISABLE"] = "1"
    return env


def get_fold_dir(dataset_name: str, configuration: str, fold: int) -> Path:
    return (
        Path(settings.DATA_DIR)
        / "results"
        / dataset_name
        / f"nnUNetTrainer__nnUNetPlans__{configuration}"
        / f"fold_{fold}"
    )


def find_latest_training_log(dataset_name: str, configuration: str, fold: int) -> Optional[Path]:
    """Return the most recently modified training_log_*.txt in the fold directory, or None."""
    fold_dir = get_fold_dir(dataset_name, configuration, fold)
    logs = sorted(fold_dir.glob("training_log_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def get_validation_summary_path(dataset_name: str, configuration: str, fold: int) -> Path:
    return (
        Path(settings.DATA_DIR)
        / "results"
        / dataset_name
        / f"nnUNetTrainer__nnUNetPlans__{configuration}"
        / f"fold_{fold}"
        / "validation"
        / "summary.json"
    )


# ---------------------------------------------------------------------------
# Dataset setup
# ---------------------------------------------------------------------------

def setup_dataset(zip_path: str, dataset_name: str) -> None:
    """
    Extract dataset ZIP into the nnUNet directory structure.

    ZIP structure (produced by nnunet_server upload):
        Dataset###_Name/imagesTr/...
        Dataset###_Name/labelsTr/...
        Dataset###_Name/dataset.json
        Dataset###_Name/dataset_fingerprint.json   <- goes to preprocessed/
        Dataset###_Name/nnUNetPlans.json            <- goes to preprocessed/
    """
    data_dir = Path(settings.DATA_DIR)
    raw_dest = data_dir / "raw"
    preprocessed_dest = data_dir / "preprocessed" / dataset_name

    raw_dest.mkdir(parents=True, exist_ok=True)
    preprocessed_dest.mkdir(parents=True, exist_ok=True)

    plan_files = {"dataset_fingerprint.json", "nnUNetPlans.json"}
    dual_files = {"dataset.json"}  # goes to both raw/ and preprocessed/

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            basename = os.path.basename(member)
            if not basename:
                continue  # skip directory entries

            if basename in plan_files:
                dest = preprocessed_dest / basename
                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                logger.info(f"  plan file: {basename} -> {dest}")
            elif basename in dual_files:
                raw_file = raw_dest / member
                raw_file.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(raw_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                pre_file = preprocessed_dest / basename
                shutil.copy2(raw_file, pre_file)
                logger.info(f"  dual file: {basename} -> raw/ + preprocessed/")
            else:
                dest = raw_dest / member
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    logger.info(f"Dataset {dataset_name} extracted to {data_dir}")


def _read_num_training(dataset_name: str) -> int:
    """Read numTraining from dataset.json."""
    ds_json = Path(settings.DATA_DIR) / "raw" / dataset_name / "dataset.json"
    try:
        with open(ds_json) as f:
            return json.load(f).get("numTraining", 0)
    except Exception:
        return 0


def is_dataset_downloaded(dataset_id: str) -> bool:
    """Return True if the dataset ZIP is already on disk."""
    zip_path = Path(settings.DATA_DIR) / "downloads" / f"{dataset_id}.zip"
    return zip_path.exists() and zip_path.stat().st_size > 0


PREPROCESSING_FLAG = "preprocessing_completed.txt"


def is_preprocessing_done(dataset_name: str) -> bool:
    """Return True if the preprocessing completion flag file exists."""
    flag = Path(settings.DATA_DIR) / "preprocessed" / dataset_name / PREPROCESSING_FLAG
    return flag.exists()


# ---------------------------------------------------------------------------
# Preprocessing via SLURM
# ---------------------------------------------------------------------------

def run_preprocess(
    job_id: str,
    dataset_name: str,
    progress_callback: Callable,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """
    Submit nnUNetv2_preprocess as a SLURM job and wait for it to finish.
    Monitors the SLURM stdout log to parse "Preprocessing case" lines for progress.
    Calls progress_callback(total_images, done_images, mean_time_s).
    """
    dataset_num = get_dataset_num(dataset_name)
    total_images = _read_num_training(dataset_name)
    logger.info(f"Submitting preprocessing SLURM job for {dataset_name} (num={dataset_num}, total={total_images})")

    log_dir = Path(settings.DATA_DIR) / "logs" / job_id / "preprocess"
    log_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(settings.DATA_DIR) / "slurm_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"{job_id}_preprocess.sh"

    slurm.write_preprocess_script(
        script_path=script_path,
        dataset_num=dataset_num,
        log_dir=log_dir,
        data_dir=settings.DATA_DIR,
        conda_env=settings.CONDA_ENV,
    )
    slurm_job_id = slurm.sbatch(str(script_path))
    logger.info(f"Preprocessing submitted as SLURM job {slurm_job_id}")

    slurm_log = log_dir / f"slurm_{slurm_job_id}.log"
    start_time = time.time()
    done_event = threading.Event()

    def monitor():
        done_images = 0
        last_report = time.time()
        seen_lines = 0

        while not done_event.is_set():
            done_event.wait(10)

            if not slurm_log.exists():
                continue
            try:
                content = slurm_log.read_text(errors="replace")
            except Exception:
                continue

            lines = content.splitlines()
            new_lines = lines[seen_lines:]
            seen_lines = len(lines)

            for line in new_lines:
                if "Preprocessing case" in line or "preprocessing case" in line.lower():
                    done_images += 1
                    now = time.time()
                    mean_s = (now - start_time) / done_images
                    if now - last_report >= 10:
                        try:
                            progress_callback(total_images, done_images, mean_s)
                        except Exception as e:
                            logger.warning(f"Preprocessing progress callback failed: {e}")
                        last_report = now

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    try:
        slurm.wait_for_slurm_job(slurm_job_id, cancel_event)
    finally:
        done_event.set()
        monitor_thread.join(timeout=10)

    # Final progress report
    try:
        elapsed = time.time() - start_time
        progress_callback(total_images, total_images, elapsed / total_images if total_images else None)
    except Exception:
        pass

    # Write flag so future jobs skip preprocessing for this dataset
    flag = Path(settings.DATA_DIR) / "preprocessed" / dataset_name / PREPROCESSING_FLAG
    flag.write_text(f"Preprocessing completed for job {job_id} (SLURM job {slurm_job_id})\n")
    logger.info(f"Preprocessing complete for {dataset_name}")


# ---------------------------------------------------------------------------
# Training via SLURM
# ---------------------------------------------------------------------------

def run_train_fold(
    job_id: str,
    dataset_name: str,
    configuration: str,
    fold: int,
    progress_callback: Callable,
    log_upload_callback: Callable,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """
    Submit nnUNetv2_train for a single fold as a SLURM job and wait for it to finish.
    Monitors the nnUNet training_log_*.txt file in a background thread while waiting.
    Calls progress_callback(fold, epoch, lr, train_loss, val_loss, pseudo_dice, epoch_time_s).
    Calls log_upload_callback(fold, text) periodically.
    """
    dataset_num = get_dataset_num(dataset_name)
    logger.info(f"Submitting training SLURM job: {dataset_name} {configuration} fold {fold}")

    log_dir = Path(settings.DATA_DIR) / "logs" / job_id / f"fold_{fold}"
    log_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(settings.DATA_DIR) / "slurm_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / f"{job_id}_train_{configuration}_fold{fold}.sh"

    slurm.write_train_script(
        script_path=script_path,
        dataset_num=dataset_num,
        configuration=configuration,
        fold=fold,
        log_dir=log_dir,
        data_dir=settings.DATA_DIR,
        conda_env=settings.CONDA_ENV,
    )
    slurm_job_id = slurm.sbatch(str(script_path))
    logger.info(f"Training fold {fold} submitted as SLURM job {slurm_job_id}")

    stop_event = threading.Event()

    def monitor():
        last_reported_epoch = -1
        last_log_upload = time.time()

        while not stop_event.is_set():
            stop_event.wait(5)

            log_path = find_latest_training_log(dataset_name, configuration, fold)
            if log_path is None:
                continue

            try:
                content = log_path.read_text(errors="replace")
            except Exception:
                continue

            epoch_data = _parse_all_epochs(content)

            for ep_num in sorted(epoch_data.keys()):
                if ep_num > last_reported_epoch:
                    ep = epoch_data[ep_num]
                    try:
                        progress_callback(
                            fold=fold,
                            epoch=ep_num,
                            learning_rate=ep.get("learning_rate"),
                            train_loss=ep.get("train_loss"),
                            val_loss=ep.get("val_loss"),
                            pseudo_dice=ep.get("pseudo_dice"),
                            epoch_time_s=ep.get("epoch_time_s"),
                        )
                        last_reported_epoch = ep_num
                    except Exception as e:
                        logger.warning(f"Training progress callback failed: {e}")

            if time.time() - last_log_upload >= 60:
                try:
                    log_upload_callback(fold, content)
                    last_log_upload = time.time()
                except Exception as e:
                    logger.warning(f"Log upload failed: {e}")

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    try:
        slurm.wait_for_slurm_job(slurm_job_id, cancel_event)
    finally:
        stop_event.set()
        monitor_thread.join(timeout=10)

    # Final log upload
    log_path = find_latest_training_log(dataset_name, configuration, fold)
    if log_path is not None:
        try:
            log_upload_callback(fold, log_path.read_text(errors="replace"))
        except Exception as e:
            logger.warning(f"Final log upload failed: {e}")

    logger.info(f"Training fold {fold} complete")


# ---------------------------------------------------------------------------
# Parallel training: all folds submitted at once
# ---------------------------------------------------------------------------

def run_train_all_folds(
    job_id: str,
    dataset_name: str,
    configuration: str,
    folds: list,
    progress_callback: Callable,
    log_upload_callback: Callable,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """
    Submit all fold training jobs to SLURM simultaneously and monitor them in parallel.
    Waits for all folds to complete before returning.

    If any fold fails, remaining in-flight folds are cancelled and SlurmJobFailed is raised.
    If cancel_event is set (user cancellation), all folds are cancelled and JobCancelled is raised.

    Calls progress_callback(fold, epoch, lr, train_loss, val_loss, pseudo_dice, epoch_time_s).
    Calls log_upload_callback(fold, text) periodically per fold.
    """
    dataset_num = get_dataset_num(dataset_name)
    logger.info(
        f"Submitting {len(folds)} training SLURM jobs in parallel: "
        f"{dataset_name} {configuration} folds={folds}"
    )

    scripts_dir = Path(settings.DATA_DIR) / "slurm_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Submit all folds
    slurm_job_ids: dict = {}
    for fold in folds:
        log_dir = Path(settings.DATA_DIR) / "logs" / job_id / f"fold_{fold}"
        log_dir.mkdir(parents=True, exist_ok=True)
        script_path = scripts_dir / f"{job_id}_train_{configuration}_fold{fold}.sh"
        slurm.write_train_script(
            script_path=script_path,
            dataset_num=dataset_num,
            configuration=configuration,
            fold=fold,
            log_dir=log_dir,
            data_dir=settings.DATA_DIR,
            conda_env=settings.CONDA_ENV,
        )
        slurm_job_ids[fold] = slurm.sbatch(str(script_path))
        logger.info(f"Training fold {fold} submitted as SLURM job {slurm_job_ids[fold]}")

    # abort_event: set when any fold fails OR user cancels — scancels all remaining folds
    abort_event = threading.Event()
    if cancel_event is not None:
        def _watch_user_cancel():
            cancel_event.wait()
            abort_event.set()
        threading.Thread(target=_watch_user_cancel, daemon=True).start()

    fold_exceptions: dict = {}
    lock = threading.Lock()

    def run_fold_waiter(fold: int, slurm_job_id: str) -> None:
        try:
            slurm.wait_for_slurm_job(slurm_job_id, abort_event)
        except Exception as e:
            with lock:
                fold_exceptions[fold] = e
            abort_event.set()  # trigger cancellation of other still-running folds
        finally:
            logger.info(f"Fold {fold} waiter finished (SLURM job {slurm_job_id})")

    stop_monitor = threading.Event()

    def run_fold_monitor(fold: int) -> None:
        last_reported_epoch = -1
        last_log_upload = time.time()
        while not stop_monitor.is_set():
            stop_monitor.wait(5)
            log_path = find_latest_training_log(dataset_name, configuration, fold)
            if log_path is None:
                continue
            try:
                content = log_path.read_text(errors="replace")
            except Exception:
                continue
            epoch_data = _parse_all_epochs(content)
            for ep_num in sorted(epoch_data.keys()):
                if ep_num > last_reported_epoch:
                    ep = epoch_data[ep_num]
                    try:
                        progress_callback(
                            fold=fold,
                            epoch=ep_num,
                            learning_rate=ep.get("learning_rate"),
                            train_loss=ep.get("train_loss"),
                            val_loss=ep.get("val_loss"),
                            pseudo_dice=ep.get("pseudo_dice"),
                            epoch_time_s=ep.get("epoch_time_s"),
                        )
                        last_reported_epoch = ep_num
                    except Exception as e:
                        logger.warning(f"Training progress callback failed fold {fold}: {e}")
            if time.time() - last_log_upload >= 60:
                try:
                    log_upload_callback(fold, content)
                    last_log_upload = time.time()
                except Exception as e:
                    logger.warning(f"Log upload failed fold {fold}: {e}")

    monitor_threads = [
        threading.Thread(target=run_fold_monitor, args=(fold,), daemon=True)
        for fold in folds
    ]
    waiter_threads = [
        threading.Thread(target=run_fold_waiter, args=(fold, slurm_job_ids[fold]), daemon=True)
        for fold in folds
    ]
    for t in monitor_threads:
        t.start()
    for t in waiter_threads:
        t.start()

    for t in waiter_threads:
        t.join()

    stop_monitor.set()
    for t in monitor_threads:
        t.join(timeout=10)

    # Final log uploads
    for fold in folds:
        log_path = find_latest_training_log(dataset_name, configuration, fold)
        if log_path is not None:
            try:
                log_upload_callback(fold, log_path.read_text(errors="replace"))
            except Exception as e:
                logger.warning(f"Final log upload failed fold {fold}: {e}")

    logger.info(f"All {len(folds)} training folds finished")

    # User cancellation takes priority
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled("Training cancelled by user")

    # Surface real failures (not secondary JobCancelled from abort cascade)
    real_failures = {f: e for f, e in fold_exceptions.items() if not isinstance(e, JobCancelled)}
    if real_failures:
        fold, exc = next(iter(real_failures.items()))
        raise SlurmJobFailed(f"Fold {fold} failed: {exc}")

    if fold_exceptions:
        fold, exc = next(iter(fold_exceptions.items()))
        raise exc


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def _parse_all_epochs(log_content: str) -> dict:
    """
    Parse all complete epoch blocks from nnUNet training log.
    Returns dict of {epoch_num: {epoch, learning_rate, train_loss, val_loss, pseudo_dice, epoch_time_s}}.
    An epoch block is complete when epoch_time_s is found.
    """
    epochs = {}
    current = {}

    for line in log_content.splitlines():
        m = _RE_EPOCH.search(line)
        if m:
            current = {"epoch": int(m.group(1))}
            continue
        if not current:
            continue
        m = _RE_LR.search(line)
        if m:
            current["learning_rate"] = float(m.group(1))
            continue
        m = _RE_TRAIN_LOSS.search(line)
        if m:
            current["train_loss"] = float(m.group(1))
            continue
        m = _RE_VAL_LOSS.search(line)
        if m:
            current["val_loss"] = float(m.group(1))
            continue
        m = _RE_DICE.search(line)
        if m:
            vals = [v.strip() for v in m.group(1).split(",")]
            try:
                def _to_float(s):
                    pm = _RE_PARENS.search(s)
                    return float(pm.group(1).strip() if pm else s)
                current["pseudo_dice"] = json.dumps([_to_float(v) for v in vals])
            except Exception:
                current["pseudo_dice"] = json.dumps(vals)
            continue
        m = _RE_EPOCH_TIME.search(line)
        if m:
            current["epoch_time_s"] = float(m.group(1))
            ep_num = current.get("epoch", -1)
            if ep_num >= 0:
                epochs[ep_num] = dict(current)
            current = {}

    return epochs


# ---------------------------------------------------------------------------
# Validation results
# ---------------------------------------------------------------------------

def read_validation_result(dataset_name: str, configuration: str, fold: int) -> Optional[str]:
    """Read fold validation summary.json and return as JSON string, or None if missing."""
    path = get_validation_summary_path(dataset_name, configuration, fold)
    if not path.exists():
        logger.warning(f"Validation summary not found: {path}")
        return None
    return path.read_text()


# ---------------------------------------------------------------------------
# Model export (runs locally, not via SLURM — fast operation)
# ---------------------------------------------------------------------------

def export_model(dataset_name: str, configuration: str) -> Path:
    """
    Export trained model to ZIP using nnUNetv2_export_model_to_zip.
    Runs locally (not via SLURM) since it is a fast post-processing step.
    Returns the path to the created ZIP file.
    """
    import subprocess
    output_dir = Path(settings.DATA_DIR) / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = output_dir / f"{dataset_name}_{configuration}.zip"

    if output_zip.exists():
        output_zip.unlink()

    env = get_nnunet_env()
    dataset_num = get_dataset_num(dataset_name)

    # Activate conda in the same shell command
    conda_base_result = subprocess.run(
        ["conda", "info", "--base"], capture_output=True, text=True
    )
    conda_base = conda_base_result.stdout.strip() if conda_base_result.returncode == 0 else ""
    conda_profile = f"{conda_base}/etc/profile.d/conda.sh" if conda_base else ""

    activate = (
        f'source "{conda_profile}" && conda activate "{settings.CONDA_ENV}" && '
        if conda_profile
        else ""
    )
    cmd = (
        f"{activate}"
        f"nnUNetv2_export_model_to_zip "
        f'-d "{dataset_num}" '
        f'-c "{configuration}" '
        f'-o "{output_zip}" '
        f"--not_strict"
    )

    logger.info(f"Exporting model: {dataset_name} {configuration} -> {output_zip}")
    result = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Model export failed (exit {result.returncode}):\n{result.stderr}"
        )
    if not output_zip.exists():
        raise RuntimeError(f"Export succeeded but ZIP not found at {output_zip}")

    logger.info(f"Model exported: {output_zip} ({output_zip.stat().st_size / 1024 / 1024:.1f} MB)")
    return output_zip
