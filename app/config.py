from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Dashboard connection
    DASHBOARD_URL: str = "http://localhost:9333"
    DASHBOARD_API_KEY: str = "changeme"

    # Notifications (optional)
    GOOGLE_CHAT_WEBHOOK_URL: str = ""

    # Worker identity (reported at registration)
    WORKER_NAME: str = "worker-slurm01"
    WORKER_HOSTNAME: str = ""
    GPU_NAME: str = ""
    GPU_MEMORY_GB: float = 0.0
    CPU_CORES: int = 0
    SYSTEM_MEMORY_GB: float = 0.0

    # Timing
    POLL_INTERVAL_S: int = 30       # seconds between job polls
    HEARTBEAT_INTERVAL_S: int = 60  # seconds between heartbeats

    # Paths
    DATA_DIR: str = "/data/nnunet_trainer_data"
    CONDA_ENV: str = "nnunet_trainer"

    # nnUNet
    NUM_PREPROCESSING_WORKERS: int = 8

    # SLURM — preprocessing (CPU-only job on b40x4 partition)
    SLURM_PARTITION_PREPROCESS: str = "b40x4"
    SLURM_CPUS_PREPROCESS: int = 16
    SLURM_MEM_PREPROCESS: str = "128G"
    SLURM_TIME_PREPROCESS: str = "8:00:00"

    # SLURM — training (GPU job on b40x4-long partition)
    SLURM_PARTITION_TRAIN: str = "b40x4-long"
    SLURM_CPUS_TRAIN: int = 16
    SLURM_MEM_TRAIN: str = "128G"
    SLURM_TIME_TRAIN: str = "2-00:00:00"
    SLURM_GPUS_TRAIN: int = 1

    # SLURM — email notifications (optional, leave blank to disable)
    SLURM_MAIL_USER: str = ""
    SLURM_MAIL_TYPE: str = "ALL"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()
