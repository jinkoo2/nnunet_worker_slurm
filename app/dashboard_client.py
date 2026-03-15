import os
import logging
import time
import requests
from .config import settings

logger = logging.getLogger(__name__)

_RETRY_ON = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


class DashboardClient:
    def __init__(self):
        self.base = settings.DASHBOARD_URL.rstrip("/")
        self.headers = {"X-Api-Key": settings.DASHBOARD_API_KEY}

    def _request_with_retry(self, fn, max_retries=3):
        """Call fn() and retry on transient network errors or 5xx responses."""
        for attempt in range(max_retries + 1):
            try:
                r = fn()
                if r.status_code >= 500 and attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Dashboard returned {r.status_code}, retrying in {wait}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r
            except _RETRY_ON as e:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Dashboard unreachable ({e.__class__.__name__}: {e}), "
                        f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait)
                else:
                    raise

    def _get(self, path, params=None):
        r = self._request_with_retry(
            lambda: requests.get(
                f"{self.base}{path}", headers=self.headers, params=params, timeout=30
            )
        )
        return r.json()

    def _post(self, path, json=None, data=None, files=None, timeout=30):
        headers = dict(self.headers)
        if files:
            headers.pop("Content-Type", None)
        r = self._request_with_retry(
            lambda: requests.post(
                f"{self.base}{path}",
                headers=headers,
                json=json,
                data=data,
                files=files,
                timeout=timeout,
            )
        )
        return r.json()

    def _put(self, path, json=None):
        r = self._request_with_retry(
            lambda: requests.put(
                f"{self.base}{path}", headers=self.headers, json=json, timeout=30
            )
        )
        return r.json()

    # -------------------------------------------------------------------------
    # Workers
    # -------------------------------------------------------------------------

    def register_worker(self) -> dict:
        return self._post(
            "/api/workers/register",
            json={
                "name": settings.WORKER_NAME,
                "hostname": settings.WORKER_HOSTNAME or None,
                "cpu_cores": settings.CPU_CORES or None,
                "gpu_memory_gb": settings.GPU_MEMORY_GB or None,
                "gpu_name": settings.GPU_NAME or None,
                "system_memory_gb": settings.SYSTEM_MEMORY_GB or None,
            },
        )

    def heartbeat(self, worker_id: str, status: str = "online") -> None:
        self._post(f"/api/workers/{worker_id}/heartbeat", json={"status": status})

    def post_log(self, worker_id: str, worker_name: str, level: str, message: str) -> None:
        self._post("/api/logs/", json={
            "worker_id": worker_id,
            "worker_name": worker_name,
            "level": level,
            "message": message,
        })

    # -------------------------------------------------------------------------
    # Jobs
    # -------------------------------------------------------------------------

    def get_pending_jobs(self, worker_id: str) -> list:
        return self._get("/api/jobs/", params={"worker_id": worker_id, "status": "pending"})

    def update_job_status(self, job_id: str, status: str, error_message: str = None) -> None:
        body = {"status": status}
        if error_message is not None:
            body["error_message"] = error_message
        self._put(f"/api/jobs/{job_id}/status", json=body)

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------

    def get_job(self, job_id: str) -> dict:
        return self._get(f"/api/jobs/{job_id}")

    def get_dataset(self, dataset_id: str) -> dict:
        return self._get(f"/api/datasets/{dataset_id}")

    def download_dataset(self, dataset_id: str, dest_path: str) -> None:
        logger.info(f"Downloading dataset {dataset_id} → {dest_path}")
        r = requests.get(
            f"{self.base}/api/datasets/{dataset_id}/download",
            headers=self.headers,
            stream=True,
            timeout=3600,
        )
        r.raise_for_status()
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
        logger.info(f"Download complete: {downloaded / 1024 / 1024:.1f} MB")

    # -------------------------------------------------------------------------
    # Progress reporting
    # -------------------------------------------------------------------------

    def report_preprocessing_progress(
        self, job_id: str, total_images: int, done_images: int, mean_time_s: float = None
    ) -> None:
        self._post(
            f"/api/jobs/{job_id}/preprocessing_progress",
            json={
                "total_images": total_images,
                "done_images": done_images,
                "mean_time_per_image_s": mean_time_s,
            },
        )

    def report_training_progress(
        self, job_id: str, fold: int, epoch: int,
        learning_rate: float = None, train_loss: float = None,
        val_loss: float = None, pseudo_dice: str = None, epoch_time_s: float = None,
    ) -> None:
        self._post(
            f"/api/jobs/{job_id}/training_progress",
            json={
                "fold": fold,
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "pseudo_dice": pseudo_dice,
                "epoch_time_s": epoch_time_s,
            },
        )

    def report_validation_result(self, job_id: str, fold: int, summary_json: str) -> None:
        self._post(
            f"/api/jobs/{job_id}/validation_result",
            json={"fold": fold, "summary_json": summary_json},
        )

    def upload_log(self, job_id: str, fold: int, text: str) -> None:
        r = requests.post(
            f"{self.base}/api/jobs/{job_id}/log/{fold}",
            headers=self.headers,
            data=text.encode("utf-8"),
            timeout=60,
        )
        r.raise_for_status()

    def upload_model(self, job_id: str, zip_path: str, chunk_size_mb: int = 50) -> dict:
        """Upload model ZIP in chunks to work around ingress body-size limits."""
        chunk_size = chunk_size_mb * 1024 * 1024
        file_size = os.path.getsize(zip_path)
        total_chunks = (file_size + chunk_size - 1) // chunk_size

        logger.info(f"Chunked model upload: {zip_path} ({file_size / 1024 / 1024:.1f} MB, {total_chunks} chunks)")

        resp = self._post(f"/api/jobs/{job_id}/model/upload/init", json={
            "total_chunks": total_chunks,
            "total_size": file_size,
        })
        upload_id = resp["upload_id"]
        logger.info(f"Model upload session: {upload_id}")

        with open(zip_path, "rb") as f:
            for i in range(total_chunks):
                data = f.read(chunk_size)
                r = self._request_with_retry(
                    lambda: requests.post(
                        f"{self.base}/api/jobs/{job_id}/model/upload/{upload_id}/chunk/{i}",
                        headers=self.headers,
                        data=data,
                        timeout=300,
                    )
                )
                logger.info(f"  chunk {i + 1}/{total_chunks} uploaded")

        logger.info("Completing model upload...")
        return self._post(f"/api/jobs/{job_id}/model/upload/{upload_id}/complete", timeout=120)
