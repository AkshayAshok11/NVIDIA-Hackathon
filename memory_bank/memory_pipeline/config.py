"""
Central config for the Timebox memory pipeline.

All models run LOCALLY on the DGX Spark (Acer Veriton GN100) — no cloud calls,
per the privacy requirement (memories never leave the device).

Before running the pipeline, these three services must be up on the Veriton:

  1. VLM (captioning)   -> default: http://localhost:8001
  2. Embed (embeddings)  -> default: http://localhost:8002
  3. Lightning (reasoning/orchestration) -> default: http://localhost:8003

Adjust the *_BASE_URL values below to match however you end up deploying each
one (Ollama, NIM container, etc.) — ports are placeholders until you confirm
the real ones during setup.
"""

import os
from pathlib import Path

# --- Model endpoints (all local) ---------------------------------------
VLM_BASE_URL = os.environ.get("TIMEBOX_VLM_URL", "http://localhost:8001/v1")
VLM_MODEL_NAME = os.environ.get("TIMEBOX_VLM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")

EMBED_BASE_URL = os.environ.get("TIMEBOX_EMBED_URL", "http://localhost:8002/v1")
EMBED_MODEL_NAME = os.environ.get("TIMEBOX_EMBED_MODEL", "nvidia/nemotron-3-embed-1b")

LIGHTNING_BASE_URL = os.environ.get("TIMEBOX_LIGHTNING_URL", "http://localhost:11434/v1")
LIGHTNING_MODEL_NAME = os.environ.get(
    "TIMEBOX_LIGHTNING_MODEL", "nemotron-3.5-lightning"
)

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
