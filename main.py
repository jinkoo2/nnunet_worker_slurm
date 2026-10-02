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
import os
import socket
import sys
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("main")


def _install_dashboard_dns_override() -> None:
    """Temporary workaround when cluster DNS cannot resolve the dashboard host.

    Set DASHBOARD_RESOLVE_IP to a reachable A record (e.g. from ``dig @8.8.8.8``).
    Keeps DASHBOARD_URL hostname for TLS/SNI; only forces the IP used to connect.
    Remove this once IT restores resolution for *.myphysics.net.
    """
    ip = os.environ.get("DASHBOARD_RESOLVE_IP", "").strip()
    if not ip:
        # Also honor value from the selected ENV_FILE before Settings loads.
        env_file = os.environ.get("ENV_FILE", ".env")
        if os.path.isfile(env_file):
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DASHBOARD_RESOLVE_IP="):
                        ip = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not ip:
        return

    host = ""
    url = os.environ.get("DASHBOARD_URL", "").strip()
    if not url and os.path.isfile(os.environ.get("ENV_FILE", ".env")):
        with open(os.environ.get("ENV_FILE", ".env"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DASHBOARD_URL="):
                    url = line.split("=", 1)[1].strip()
                    break
    if url:
        host = urlparse(url).hostname or ""
    if not host:
        host = "nnunet-dashboard-1.apps.myphysics.net"

    _orig = socket.getaddrinfo

    def _getaddrinfo(name, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
        if name == host:
            name = ip
        return _orig(name, port, family, type, proto, flags)

    socket.getaddrinfo = _getaddrinfo  # type: ignore[assignment]
    log.warning(
        "DASHBOARD_RESOLVE_IP=%s — forcing %s → %s (remove when cluster DNS works again)",
        ip,
        host,
        ip,
    )


_install_dashboard_dns_override()

from app.worker import run  # noqa: E402

if __name__ == "__main__":
    run()
