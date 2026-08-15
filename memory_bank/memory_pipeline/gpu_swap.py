"""
Both the VLM and Embed NIM containers can now run simultaneously, as long as
the VLM is configured with NIM_KV_CACHE_PERCENT=0.6 (found by tracing NIM's
own source — the documented --gpu-memory-utilization flag and its env-var
equivalents don't reach this NIM's actual engine config; NIM_KV_CACHE_PERCENT
is the real override). At 0.6, the VLM uses ~71-73GB, leaving enough of the
GPU's 121GB for Embed (~3.6GB) and other GPU work to coexist.

This module now just confirms the needed service is healthy — it no longer
reflexively stops the other container first. If starting a container fails
(e.g. genuine memory exhaustion from something else running), it falls back
to the old swap behavior as a safety net, rather than assuming the fix always
holds under any load.
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


def _wait_healthy(url: str, container_name: str, timeout_sec: int = STARTUP_TIMEOUT_SEC) -> bool:
    """Poll until healthy or timeout. Returns True/False instead of raising,
    so callers can decide whether to fall back rather than crash outright."""
    start = time.time()
    while time.time() - start < timeout_sec:
        if _is_healthy(url):
            return True
        time.sleep(POLL_INTERVAL_SEC)
    return False


def _ensure_active(target_url: str, target_container: str, other_container: str) -> None:
    if _is_healthy(target_url):
        return  # already up, nothing to do — the common case now

    print(f"      ({target_container} not running — starting it)")
    _run(["docker", "start", target_container])

    if _wait_healthy(target_url, target_container):
        return  # started fine alongside whatever else is running — no swap needed

    # Didn't come up in time — likely genuine memory pressure (heavy GPU use
    # elsewhere). Fall back to the old behavior: stop the other container to
    # free room, then retry.
    print(
        f"      ({target_container} didn't start within {STARTUP_TIMEOUT_SEC}s — "
        f"stopping {other_container} to free memory and retrying)"
    )
    _run(["docker", "stop", other_container])
    _run(["docker", "start", target_container])
    if not _wait_healthy(target_url, target_container):
        raise TimeoutError(
            f"{target_container} still not healthy after stopping {other_container}. "
            f"Check `docker logs {target_container}` and `nvidia-smi` — this may need "
            f"manual intervention (e.g. someone else's GPU usage)."
        )


def ensure_vlm_active() -> None:
    """Start VLM if it's not already up. Tries alongside Embed first; only
    stops Embed if the VLM genuinely can't fit."""
    _ensure_active(VLM_HEALTH_URL, VLM_CONTAINER, EMBED_CONTAINER)


def ensure_embed_active() -> None:
    """Start Embed if it's not already up. Tries alongside the VLM first;
    only stops the VLM if Embed genuinely can't fit."""
    _ensure_active(EMBED_HEALTH_URL, EMBED_CONTAINER, VLM_CONTAINER)