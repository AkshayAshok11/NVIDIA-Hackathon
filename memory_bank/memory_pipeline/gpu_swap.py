"""
Both the VLM and Embed NIM containers can run simultaneously when the VLM is
configured with NIM_KV_CACHE_PERCENT=0.6. At 0.6, the VLM uses ~71-73GB,
leaving enough of the GPU's 121GB for Embed (~3.6GB) and other GPU work.

This module starts the needed service if it is down. It will not stop a
healthy sibling just because the target is slow to become ready — that was
killing a working VLM after captioning whenever Embed took too long.
"""

from __future__ import annotations

import subprocess
import time

import httpx

VLM_CONTAINER = "nemotron-omni-vlm"
EMBED_CONTAINER = "nemotron-3-embed-1b"

VLM_HEALTH_URL = "http://localhost:8001/v1/models"
EMBED_HEALTH_URL = "http://localhost:8002/v1/models"

VLM_STARTUP_TIMEOUT_SEC = 600  # VLM cold start can take several minutes even with cached weights
EMBED_STARTUP_TIMEOUT_SEC = 180
POLL_INTERVAL_SEC = 3


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _is_healthy(url: str) -> bool:
    try:
        resp = httpx.get(url, timeout=2.0)
        return resp.status_code == 200
    except httpx.TransportError:
        # Covers ConnectError, TimeoutException, ReadError, and other
        # transport-level failures that happen while a container is mid
        # start/stop — any of these just means "not ready yet, keep polling."
        return False


def _wait_healthy(url: str, container_name: str, timeout_sec: int) -> bool:
    """Poll until healthy, the container exits, or timeout."""
    start = time.time()
    while time.time() - start < timeout_sec:
        if _is_healthy(url):
            return True
        if not _container_running(container_name):
            return False
        time.sleep(POLL_INTERVAL_SEC)
    return False


def _ensure_active(
    target_url: str,
    target_container: str,
    other_container: str,
    other_url: str,
    timeout_sec: int,
) -> None:
    if _is_healthy(target_url):
        return

    print(f"      ({target_container} not running — starting it)")
    _run(["docker", "start", target_container])

    if _wait_healthy(target_url, target_container, timeout_sec):
        return

    if _container_running(target_container):
        raise TimeoutError(
            f"{target_container} is still running but not healthy after {timeout_sec}s. "
            f"Leaving {other_container} alone. Check `docker logs {target_container}`."
        )

    # Target exited during startup. Retry once without touching a healthy sibling.
    print(f"      ({target_container} exited during startup — restarting it)")
    _run(["docker", "start", target_container])
    if _wait_healthy(target_url, target_container, timeout_sec):
        return

    if _is_healthy(other_url):
        raise TimeoutError(
            f"{target_container} still not healthy after a restart. "
            f"Not stopping healthy {other_container}. "
            f"Check `docker logs {target_container}` and `nvidia-smi`."
        )

    print(
        f"      ({target_container} still down and {other_container} is not healthy — "
        f"stopping {other_container} and retrying)"
    )
    subprocess.run(["docker", "stop", other_container], capture_output=True, text=True)
    _run(["docker", "start", target_container])
    if not _wait_healthy(target_url, target_container, timeout_sec):
        raise TimeoutError(
            f"{target_container} still not healthy after stopping {other_container}. "
            f"Check `docker logs {target_container}` and `nvidia-smi`."
        )


def ensure_vlm_active() -> None:
    """Start VLM if it's not already up. Does not stop a healthy Embed container."""
    _ensure_active(
        VLM_HEALTH_URL,
        VLM_CONTAINER,
        EMBED_CONTAINER,
        EMBED_HEALTH_URL,
        VLM_STARTUP_TIMEOUT_SEC,
    )


def ensure_embed_active() -> None:
    """Start Embed if it's not already up. Does not stop a healthy VLM container."""
    _ensure_active(
        EMBED_HEALTH_URL,
        EMBED_CONTAINER,
        VLM_CONTAINER,
        VLM_HEALTH_URL,
        EMBED_STARTUP_TIMEOUT_SEC,
    )
