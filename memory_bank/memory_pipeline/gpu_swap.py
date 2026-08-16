"""
GPU memory on this box isn't large enough for the VLM and Embed NIM
containers to run at full default settings simultaneously (the VLM reserves
~90% of GPU memory on startup). Rather than fight NIM's memory-utilization
flags, this swaps containers on demand: stop whichever one isn't needed,
start the one that is, wait for it to report healthy.

This adds real latency to each ingest/query call (container startup takes
time), but it's reliable, which matters more than speed for a hackathon
demo. If GPU memory headroom improves (e.g. the VLM's utilization flag gets
sorted out, or you move to a bigger box), this whole module can be deleted
and the pipeline used directly against always-on containers.
"""

from __future__ import annotations

import subprocess
import time

import httpx

VLM_CONTAINER = "nemotron-omni-vlm"
EMBED_CONTAINER = "nemotron-3-embed-1b"

VLM_HEALTH_URL = "http://localhost:8001/v1/models"
EMBED_HEALTH_URL = "http://localhost:8002/v1/models"

STARTUP_TIMEOUT_SEC = 600  # VLM cold start can take several minutes even with cached weights
POLL_INTERVAL_SEC = 3


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _is_healthy(url: str) -> bool:
    try:
        resp = httpx.get(url, timeout=2.0)
        return resp.status_code == 200
    except httpx.TransportError:
        # Covers ConnectError, TimeoutException, ReadError, and other
        # transport-level failures that happen while a container is mid
        # start/stop — any of these just means "not ready yet, keep polling."
        return False


def _wait_healthy(url: str, container_name: str) -> None:
    start = time.time()
    while time.time() - start < STARTUP_TIMEOUT_SEC:
        if _is_healthy(url):
            return
        time.sleep(POLL_INTERVAL_SEC)
    raise TimeoutError(
        f"{container_name} did not become healthy within {STARTUP_TIMEOUT_SEC}s. "
        f"Check `docker logs {container_name}`."
    )


def ensure_vlm_active() -> None:
    """Stop Embed (if running), start/confirm VLM, wait until it's serving."""
    if _is_healthy(VLM_HEALTH_URL):
        return  # already up, nothing to do
    print("      (swapping GPU: stopping Embed, starting VLM — this can take a few minutes)")
    _run(["docker", "stop", EMBED_CONTAINER])
    _run(["docker", "start", VLM_CONTAINER])
    _wait_healthy(VLM_HEALTH_URL, VLM_CONTAINER)


def ensure_embed_active() -> None:
    """Stop VLM (if running), start/confirm Embed, wait until it's serving."""
    if _is_healthy(EMBED_HEALTH_URL):
        return
    print("      (swapping GPU: stopping VLM, starting Embed)")
    _run(["docker", "stop", VLM_CONTAINER])
    _run(["docker", "start", EMBED_CONTAINER])
    _wait_healthy(EMBED_HEALTH_URL, EMBED_CONTAINER)