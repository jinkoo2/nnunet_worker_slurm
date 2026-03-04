"""
Main worker loop: register → heartbeat → poll → execute jobs.
"""
import logging
import queue
import threading
import time
from pathlib import Path

from .config import settings
from .dashboard_client import DashboardClient
from . import trainer, notifier
from .slurm import JobCancelled

logger = logging.getLogger(__name__)


class DashboardLogHandler(logging.Handler):
    """Forwards log records to the dashboard via a background queue thread."""

    def __init__(self, client: DashboardClient, worker_id: str, worker_name: str):
        super().__init__(level=logging.INFO)
        self.client = client
        self.worker_id = worker_id
        self.worker_name = worker_name
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    def emit(self, record: logging.LogRecord):
        try:
            self._queue.put_nowait((record.levelname, self.format(record)))
        except Exception:
            pass

    def _send_loop(self):
        while True:
            try:
                level, message = self._queue.get(timeout=2)
                try:
                    self.client.post_log(self.worker_id, self.worker_name, level, message)
                except Exception:
                    pass
                self._queue.task_done()
            except queue.Empty:
                continue


_job_running = threading.Event()


def run():
    client = DashboardClient()

    worker_id = _register_with_retry(client)

    dash_handler = DashboardLogHandler(client, worker_id, settings.WORKER_NAME)
    dash_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logging.getLogger("app").addHandler(dash_handler)

    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(client, worker_id), daemon=True
    )
    hb_thread.start()

    logger.info(f"Worker {worker_id!r} ready. Polling every {settings.POLL_INTERVAL_S}s...")

    while True:
        try:
            jobs = client.get_pending_jobs(worker_id)
            if jobs:
                logger.info(f"Found {len(jobs)} pending job(s). Starting first.")
                _execute_job(client, jobs[0])
            else:
                logger.debug("No pending jobs.")
        except Exception as e:
            logger.error(f"Error in poll loop: {e}", exc_info=True)
            notifier.on_exception(settings.WORKER_NAME, "poll loop", str(e))

        time.sleep(settings.POLL_INTERVAL_S)


def _register_with_retry(client: DashboardClient, max_attempts: int = 10) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            worker = client.register_worker()
            worker_id = worker["id"]
            logger.info(
                f"Registered as '{settings.WORKER_NAME}' (id={worker_id[:8]}…) "
                f"at {settings.DASHBOARD_URL}"
            )
            notifier.on_registered(settings.WORKER_NAME, worker_id, settings.DASHBOARD_URL)
            return worker_id
        except Exception as e:
            wait = min(30, 5 * attempt)
            logger.warning(f"Registration attempt {attempt}/{max_attempts} failed: {e}. Retrying in {wait}s...")
            if attempt == max_attempts:
                raise
            time.sleep(wait)


def _poll_cancellation(
    client: DashboardClient,
    job_id: str,
    cancel_event: threading.Event,
    interval: int = 30,
) -> None:
    """Poll the dashboard every `interval` seconds; set cancel_event if job is cancelled."""
    while not cancel_event.wait(interval):
        try:
            job = client.get_job(job_id)
            if job.get("status") == "cancelled":
                logger.info(f"Job {job_id[:8]}… cancellation detected — stopping SLURM jobs")
                cancel_event.set()
                break
        except Exception as e:
            logger.warning(f"Cancellation poll failed: {e}")


def _heartbeat_loop(client: DashboardClient, worker_id: str):
    while True:
        try:
            status = "busy" if _job_running.is_set() else "online"
            client.heartbeat(worker_id, status)
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
        time.sleep(settings.HEARTBEAT_INTERVAL_S)


def _execute_job(client: DashboardClient, job: dict):
    job_id = job["id"]
    dataset_id = job["dataset_id"]
    configuration = job["configuration"]

    logger.info(f"=== Job {job_id[:8]}… | dataset={dataset_id[:8]}… | config={configuration} ===")
    _job_running.set()
    wn = settings.WORKER_NAME

    cancel_event = threading.Event()
    cancel_thread = threading.Thread(
        target=_poll_cancellation, args=(client, job_id, cancel_event), daemon=True
    )
    cancel_thread.start()

    try:
        # 1. Acknowledge job
        client.update_job_status(job_id, "assigned")

        # 2. Get dataset metadata
        dataset_info = client.get_dataset(dataset_id)
        dataset_name = dataset_info["name"]
        logger.info(f"Dataset name: {dataset_name}")
        notifier.on_job_start(wn, job_id, dataset_name, configuration)

        # 3. Download ZIP (skip if already on disk)
        data_dir = Path(settings.DATA_DIR)
        downloads_dir = data_dir / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        zip_path = downloads_dir / f"{dataset_id}.zip"

        if trainer.is_dataset_downloaded(dataset_id):
            mb = zip_path.stat().st_size / 1024 / 1024
            logger.info(f"Dataset ZIP already on disk ({mb:.1f} MB), skipping download")
        else:
            notifier.on_download_start(wn, job_id, dataset_name)
            client.download_dataset(dataset_id, str(zip_path))
            mb = zip_path.stat().st_size / 1024 / 1024
            notifier.on_download_complete(wn, job_id, mb)

        # 4. Extract to nnUNet directory layout (skip if raw data already present)
        raw_dir = data_dir / "raw" / dataset_name
        if raw_dir.exists():
            logger.info(f"Raw dataset already extracted to {raw_dir}, skipping extraction")
        else:
            trainer.setup_dataset(str(zip_path), dataset_name)

        # 5. Preprocess via SLURM (skip if already done)
        client.update_job_status(job_id, "preprocessing")
        if trainer.is_preprocessing_done(dataset_name):
            logger.info(f"Preprocessing already done for {dataset_name}, skipping")
            notifier.on_preprocess_complete(wn, job_id)
        else:
            num_images = trainer._read_num_training(dataset_name)
            notifier.on_preprocess_start(wn, job_id, dataset_name, num_images)

            def preprocess_progress(total, done, mean_s):
                client.report_preprocessing_progress(job_id, total, done, mean_s)

            trainer.run_preprocess(job_id, dataset_name, preprocess_progress, cancel_event)
            notifier.on_preprocess_complete(wn, job_id)

        # 6. Train all 5 folds via SLURM (one job per fold)
        client.update_job_status(job_id, "training")

        for fold in range(5):
            logger.info(f"--- Fold {fold}/4 ---")
            notifier.on_fold_start(wn, job_id, fold)

            def make_progress_cb(f):
                def cb(fold, epoch, learning_rate, train_loss, val_loss, pseudo_dice, epoch_time_s):
                    client.report_training_progress(
                        job_id, fold, epoch,
                        learning_rate=learning_rate,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        pseudo_dice=pseudo_dice,
                        epoch_time_s=epoch_time_s,
                    )
                return cb

            def make_log_cb(f):
                def cb(fold, text):
                    client.upload_log(job_id, fold, text)
                return cb

            trainer.run_train_fold(
                job_id,
                dataset_name,
                configuration,
                fold,
                progress_callback=make_progress_cb(fold),
                log_upload_callback=make_log_cb(fold),
                cancel_event=cancel_event,
            )
            notifier.on_fold_complete(wn, job_id, fold)

            summary = trainer.read_validation_result(dataset_name, configuration, fold)
            if summary:
                client.report_validation_result(job_id, fold, summary)
                logger.info(f"Validation result reported for fold {fold}")
            else:
                logger.warning(f"No validation summary found for fold {fold}")

        # 7. Export + upload model (runs locally)
        client.update_job_status(job_id, "uploading")
        notifier.on_export_start(wn, job_id)
        model_zip = trainer.export_model(dataset_name, configuration)
        client.upload_model(job_id, str(model_zip))
        logger.info("Model uploaded.")
        notifier.on_upload_complete(wn, job_id)

        # 8. Done
        client.update_job_status(job_id, "done")
        logger.info(f"=== Job {job_id[:8]}… DONE ===")
        notifier.on_job_done(wn, job_id)

    except JobCancelled:
        logger.info(f"Job {job_id[:8]}… stopped — cancelled by user")
        notifier.on_exception(wn, job_id, "cancelled by user")
        # Status is already 'cancelled' in the DB — do not overwrite
    except Exception as e:
        logger.error(f"Job {job_id[:8]}… FAILED: {e}", exc_info=True)
        notifier.on_error(wn, job_id, str(e))
        try:
            client.update_job_status(job_id, "failed", error_message=str(e))
        except Exception:
            pass
    finally:
        cancel_event.set()  # stop the cancellation poll thread
        _job_running.clear()
