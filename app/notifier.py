"""
Google Chat webhook notifications. All calls are fire-and-forget;
failures are logged as warnings and never propagate to the caller.
"""
import logging
import threading
import requests
from .config import settings

logger = logging.getLogger(__name__)


def notify(message: str) -> None:
    """Send a Google Chat notification in a background thread (non-blocking)."""
    url = settings.GOOGLE_CHAT_WEBHOOK_URL
    if not url:
        return
    threading.Thread(target=_send, args=(url, message), daemon=True).start()


def _send(url: str, message: str) -> None:
    try:
        r = requests.post(url, json={"text": message}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        logger.warning(f"Notification failed: {e}")


def _tag(worker_name: str) -> str:
    return f"[{worker_name}]"


def on_registered(worker_name: str, worker_id: str, dashboard_url: str) -> None:
    notify(f"{_tag(worker_name)} Worker registered (id={worker_id[:8]}...) at {dashboard_url}")


def on_job_start(worker_name: str, job_id: str, dataset_name: str, configuration: str) -> None:
    notify(f"{_tag(worker_name)} Job {job_id[:8]}... started — dataset={dataset_name}, config={configuration}")


def on_download_start(worker_name: str, job_id: str, dataset_name: str) -> None:
    notify(f"{_tag(worker_name)} Job {job_id[:8]}... downloading dataset {dataset_name}...")


def on_download_complete(worker_name: str, job_id: str, mb: float) -> None:
    notify(f"{_tag(worker_name)} Job {job_id[:8]}... download complete ({mb:.1f} MB)")


def on_preprocess_start(worker_name: str, job_id: str, dataset_name: str, num_images: int) -> None:
    notify(f"{_tag(worker_name)} Job {job_id[:8]}... preprocessing started — {dataset_name} ({num_images} images)")


def on_preprocess_complete(worker_name: str, job_id: str) -> None:
    notify(f"{_tag(worker_name)} Job {job_id[:8]}... preprocessing complete")


def on_fold_start(worker_name: str, job_id: str, fold: int) -> None:
    notify(f"{_tag(worker_name)} Job {job_id[:8]}... fold {fold}/4 training started")


def on_fold_complete(worker_name: str, job_id: str, fold: int) -> None:
    notify(f"{_tag(worker_name)} Job {job_id[:8]}... fold {fold}/4 training complete")


def on_export_start(worker_name: str, job_id: str) -> None:
    notify(f"{_tag(worker_name)} Job {job_id[:8]}... exporting model...")


def on_upload_complete(worker_name: str, job_id: str) -> None:
    notify(f"{_tag(worker_name)} Job {job_id[:8]}... model uploaded — pending admin approval")


def on_job_done(worker_name: str, job_id: str) -> None:
    notify(f"{_tag(worker_name)} Job {job_id[:8]}... DONE")


def on_error(worker_name: str, job_id: str, error: str) -> None:
    notify(f"{_tag(worker_name)} Job {job_id[:8]}... FAILED: {error}")


def on_exception(worker_name: str, context: str, error: str) -> None:
    notify(f"{_tag(worker_name)} ERROR in {context}: {error}")
