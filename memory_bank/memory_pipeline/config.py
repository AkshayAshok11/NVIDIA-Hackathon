"""
Central config for the Timebox memory pipeline.

All models run LOCALLY on the DGX Spark (Acer Veriton GN100) — no cloud calls,
per the privacy requirement (memories never leave the device).

Confirmed working services on the Veriton (as of tonight's deployment):

  1. VLM (captioning)        -> Nemotron 3 Nano Omni, NIM/Docker, port 8001
  2. Embed (embeddings)       -> Nemotron 3 Embed, NIM/Docker, port 8002
  3. Lightning (reasoning)    -> Nemotron 3.5 Lightning, via Ollama, port 11434
  4. ASR (speech-to-text)     -> Parakeet TDT 0.6b v3, Docker, port 8010

VLM requires NIM_KV_CACHE_PERCENT=0.6 (not the documented --gpu-memory-utilization
flag, which this NIM silently ignores) to coexist with Embed/ASR in GPU memory —
see memory_pipeline/gpu_swap.py for the coexistence-with-fallback logic.
"""

import os
from pathlib import Path

# --- Model endpoints (all local) ---------------------------------------
VLM_BASE_URL = os.environ.get("TIMEBOX_VLM_URL", "http://localhost:8001/v1")
VLM_MODEL_NAME = os.environ.get(
    "TIMEBOX_VLM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
)

EMBED_BASE_URL = os.environ.get("TIMEBOX_EMBED_URL", "http://localhost:8002/v1")
EMBED_MODEL_NAME = os.environ.get("TIMEBOX_EMBED_MODEL", "nvidia/nemotron-3-embed-1b")

LIGHTNING_BASE_URL = os.environ.get("TIMEBOX_LIGHTNING_URL", "http://localhost:11434/v1")
LIGHTNING_MODEL_NAME = os.environ.get(
    "TIMEBOX_LIGHTNING_MODEL", "nemotron-3.5-lightning"
)

# ASR — OpenAI-compatible /v1/audio/transcriptions endpoint (Parakeet TDT).
ASR_BASE_URL = os.environ.get("TIMEBOX_ASR_URL", "http://localhost:8010/v1")
ASR_MODEL_NAME = os.environ.get("TIMEBOX_ASR_MODEL", "parakeet-tdt-0.6b-v3")

# vLLM/NIM servers using the OpenAI-compatible API generally don't need a
# real key when running locally, but the client library requires *something*
# non-empty to be passed.
LOCAL_API_KEY = "not-needed"

# --- Storage --------------------------------------------------------------
DATA_DIR = Path(os.environ.get("TIMEBOX_DATA_DIR", Path(__file__).parent.parent / "data"))
MEMORY_STORE_PATH = DATA_DIR / "memory_store.json"
FAISS_INDEX_PATH = DATA_DIR / "memory_index.faiss"
FRAMES_DIR = DATA_DIR / "frames"

EMBEDDING_DIM = 2048  # nemotron-3-embed-1b native output dimension

# --- Video processing -----------------------------------------------------
FRAME_SAMPLE_INTERVAL_SEC = 2.0  # grab a frame every N seconds for captioning
MAX_FRAMES_PER_VIDEO = 8         # cap frames sent to the VLM per memory

# --- Retrieval --------------------------------------------------------------
TOP_K_RESULTS = 3
AMBIGUOUS_SCORE_GAP = 0.05  # if top-2 results are this close, treat as ambiguous