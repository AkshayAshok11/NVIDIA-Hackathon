# Timebox — Memory Pipeline

Ingests a video of a memory, auto-captions it with a local VLM, embeds it,
and stores it in a searchable memory bank. Queries are parsed and resolved
by a local reasoning model (Lightning), which decides whether to load a
confident match or ask the user to disambiguate.

All models run **locally on the DGX Spark (Acer Veriton GN100)** — no cloud
calls, so memory content never leaves the device.

## Architecture

```
INGEST:
  video.mp4 --> [frame sampling] --> [VLM captioning] --> [Embed] --> [FAISS + metadata]
                                           ^
                                     user's note (optional, can be sparse)

QUERY:
  "revisit my 18th birthday"
       --> [Lightning: clean query] --> [Embed] --> [FAISS search]
                                                           |
                                                           v
                                          [Lightning: resolve match / ambiguous / no match]
```

| Stage | Model | Job |
|---|---|---|
| Captioning | Nemotron Nano VL (or similar VLM) | Looks at video frames, writes a descriptive caption |
| Embedding | Nemotron 3 Embed (1B) | Turns text (captions, notes, queries) into vectors |
| Reasoning | Nemotron 3.5 Lightning | Cleans queries, resolves ambiguity, decides what to load |
| Storage | FAISS (flat, cosine via normalized inner product) | Vector similarity search |

## Setup

### 1. Prerequisites on the Veriton

This code assumes three local model servers are already running,
**OpenAI-compatible**, at the URLs in `memory_pipeline/config.py` (override
via env vars — see below). Standing these up is a separate step from this
repo:

- VLM server (captioning) — e.g. Nemotron Nano VL
- Embed server — Nemotron 3 Embed
- Lightning server — Nemotron 3.5 Lightning (Ollama has "day zero" support
  for this one, which may be the fastest path)

**This script does NOT deploy those models — it only calls them.** Confirm
each is reachable before running ingest/query (see Troubleshooting below).

### 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **ARM64 note:** the Veriton's GB10 is aarch64 + CUDA 13.0. `faiss-cpu` and
> `opencv-python` generally have aarch64 wheels on PyPI, but if either fails
> to install, check for a prebuilt aarch64 wheel before falling back to a
> source build. This pipeline doesn't need GPU-accelerated FAISS — the
> memory bank is small (dozens of entries), so `faiss-cpu` is plenty fast
> and avoids another CUDA-matching headache on top of everything else.

### 3. Point at your actual model endpoints

Defaults are placeholders (`localhost:8001/8002/8003`). Override with env
vars once you know the real ports:

```bash
export TIMEBOX_VLM_URL="http://localhost:XXXX/v1"
export TIMEBOX_EMBED_URL="http://localhost:XXXX/v1"
export TIMEBOX_LIGHTNING_URL="http://localhost:XXXX/v1"
```

Also confirm the model name strings in `config.py` match whatever your
server reports at `/v1/models` — these vary by how each model gets deployed.

## Usage

**Ingest a memory:**
```bash
python3 scripts/ingest_memory.py path/to/video.mp4 --note "18th birthday party"
```

**Query the memory bank:**
```bash
python3 scripts/query_memory.py "I want to revisit my 18th birthday"
```

**Attach a 3D scene once gsplat has trained it:**
```bash
python3 scripts/ingest_memory.py path/to/video.mp4 --note "..." --scene path/to/trained_scene
```
(or update the `scene_path` field on an existing memory directly in
`data/memory_store.json` if the scene finishes training after ingestion)

## Data

Memory bank lives in `data/` (gitignored — this is personal content and
shouldn't be committed):
- `data/memory_index.faiss` — vector index
- `data/memory_store.json` — metadata (notes, captions, video/scene paths)
- `data/frames/` — extracted video frames

## Troubleshooting

**"Connection refused" on ingest/query** — the relevant model server isn't
running or isn't on the port config.py expects. Check with:
```bash
curl http://localhost:PORT/v1/models
```

**VLM caption looks wrong or empty** — check the server actually supports
image inputs (multimodal) — not all local LLM deployments do by default.

**Embedding dimension mismatch** — if you swap embedding models, update
`EMBEDDING_DIM` in `config.py` to match, and note that any *existing* FAISS
index was built for the old dimension — delete `data/memory_index.faiss`
and re-ingest if you change embedding models mid-hackathon.
