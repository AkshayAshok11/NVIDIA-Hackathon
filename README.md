# Timebox

Timebox turns a short phone video of a moment into a place you can walk back
into. Record a memory, and Timebox reconstructs it as a navigable 3D
Gaussian-splat scene, indexes it against a voice-searchable memory bank, and
lets you revisit it in VR — just ask for it out loud and step through a
portal into the reconstructed scene, with an AI guide alongside you.

Built for an NVIDIA hackathon, running end-to-end **locally on a DGX Spark
(Acer Veriton GN100)** — no cloud calls, so memory content never leaves the
device.

## Quick start

Assumes the four local model servers (VLM, Embed, Lightning, ASR — see
[Reproduce the demo](#reproduce-the-demo)) are already running on the
Veriton, and you're set up on the same machine or local network.

```bash
# 0. Clone with the gsplat submodule
git clone --recursive <this-repo-url> Timebox && cd Timebox
# (or, if already cloned) git submodule update --init --recursive

# 1. Memory pipeline: env + deps
cd memory_bank
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env with your real ports
set -a && source .env && set +a

# 2. Ingest a sample memory (video -> caption -> embed -> FAISS)
python3 scripts/ingest_memory.py ../AkshaySampleVideos/IMG_4745.MOV \
  --note "example memory"

# 3. Query it from the CLI
python3 scripts/query_memory.py "show me that memory"

```

For the full VR experience, open `unity/` in the Unity Editor, build for
Quest, and point it at the API server's address — see
[Reproduce the demo](#reproduce-the-demo) below.

## How it works

```
CAPTURE
  phone video ──┬──> [VGGT + gsplat] ──> 3D Gaussian-splat scene (.ply/.splat)
                └──> [VLM caption + note] ──> [embed] ──> [FAISS memory bank]

REVISIT (voice, in VR)
  "let's go back to the campfire"
       --> [ASR: transcribe] --> [Lightning: clean query] --> [FAISS search]
                                                                    |
                                                    match / ambiguous / no match
                                                                    |
                          Unity (Quest) opens a portal --> loads the matching splat scene
```

| Stage | What runs it | Job |
|---|---|---|
| 3D reconstruction | VGGT + [gsplat](https://github.com/nerfstudio-project/gsplat) | Turns a video's frames into a trained Gaussian-splat point cloud |
| Captioning | Nemotron 3 Nano Omni (VLM) | Looks at sampled frames, writes a descriptive caption |
| Embedding | Nemotron 3 Embed (1B) | Turns captions/notes/queries into vectors |
| Speech-to-text | Parakeet TDT 0.6B v3 (ASR) | Transcribes spoken queries from the Quest mic |
| Reasoning | Nemotron 3.5 Lightning | Cleans queries, resolves ambiguity, decides what to load |
| Storage | FAISS (flat, cosine) | Vector similarity search over the memory bank |
| Playback | Unity (Meta Quest / OpenXR) | VR scene with an AI hologram guide, dialogue, and portal transitions into loaded splat scenes |
| Browser fallback | Three.js WebXR viewer | View a trained splat scene in-browser (Quest or desktop), no Unity build required |

### Tech stack

- **Hardware:** NVIDIA DGX Spark (Acer Veriton GN100, GB10, aarch64 + CUDA 13.0), Meta Quest headset
- **3D reconstruction:** [VGGT](https://github.com/facebookresearch/vggt) (pose/point-cloud inference) + [gsplat](https://github.com/nerfstudio-project/gsplat) (Gaussian-splat training), PyTorch/CUDA
- **AI models (all served locally, OpenAI-compatible APIs):** Nemotron 3 Nano Omni (VLM captioning), Nemotron 3 Embed 1B (embeddings), Nemotron 3.5 Lightning (query reasoning, via Ollama), Parakeet TDT 0.6B v3 (ASR)
- **Memory pipeline backend:** Python, FastAPI + Uvicorn, FAISS (flat/cosine), OpenCV
- **VR client:** Unity (Meta XR / OpenXR packages), C#
- **Browser viewer:** Three.js, vanilla JS, served over local HTTPS

## Repo layout

```
memory_bank/       Python pipeline: ingest video -> caption -> embed -> FAISS.
                    Also the FastAPI server (scripts/api_server.py) that Unity
                    talks to for voice-driven queries. See memory_bank/README.md.

gsplat_generation/  Video -> 3D Gaussian-splat scene. VGGT for pose/point-cloud
                    inference, gsplat (tracked as a submodule under
                    scripts/gsplat_repo) for training, plus pruning/export
                    scripts to produce viewer-ready .ply/.splat files.

webxr-viewer/       Standalone Three.js WebXR viewer for trained splat scenes —
                    serves over local HTTPS so a Quest browser can enter VR
                    without a Unity build. Includes a splat decimation tool.

unity/              The Timebox VR app (Meta Quest / OpenXR). Spawns the AI
                    hologram guide, drives dialogue and voice transcription,
                    and handles portal-ring transitions into loaded memory
                    scenes. See unity/Assets/Scripts/AIInterface/.

AkshaySampleVideos/ Example source videos used to build test memories.
```

## Reproduce the demo

Each pipeline component has its own environment; see the linked docs for
full detail. Everything runs **locally** — there are no cloud API keys.
The only "credentials" involved are local URLs + model-name strings that
have to match whatever's actually deployed on your machine.

### 1. Memory pipeline (captioning, embedding, retrieval, voice API server)

Full detail in [`memory_bank/README.md`](memory_bank/README.md). This
requires four local, OpenAI-compatible model servers already running
(standing these up is a separate step from this repo — see that README's
Prerequisites section):

| Server | Default URL | Example model |
|---|---|---|
| VLM (captioning) | `http://localhost:8001/v1` | Nemotron 3 Nano Omni |
| Embed | `http://localhost:8002/v1` | Nemotron 3 Embed (1B) |
| Lightning (reasoning) | `http://localhost:11434/v1` | Nemotron 3.5 Lightning (via Ollama) |
| ASR (speech-to-text) | `http://localhost:8010/v1` | Parakeet TDT 0.6B v3 |

Configure these via environment variables — a sample file is checked in at
[`memory_bank/.env.example`](memory_bank/.env.example):

```bash
cd memory_bank
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # edit with your real ports/model names
set -a && source .env && set +a
```

```bash
# memory_bank/.env.example
TIMEBOX_VLM_URL=http://localhost:8001/v1
TIMEBOX_VLM_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
TIMEBOX_EMBED_URL=http://localhost:8002/v1
TIMEBOX_EMBED_MODEL=nvidia/nemotron-3-embed-1b
TIMEBOX_LIGHTNING_URL=http://localhost:11434/v1
TIMEBOX_LIGHTNING_MODEL=nemotron-3.5-lightning
TIMEBOX_ASR_URL=http://localhost:8010/v1
TIMEBOX_ASR_MODEL=parakeet-tdt-0.6b-v3
# TIMEBOX_DATA_DIR=./data   # optional override, defaults to memory_bank/data/
```

Sanity-check a server is up before running anything: `curl
http://localhost:PORT/v1/models`.

### 2. Gsplat generation (video → 3D scene)

See [`gsplat_generation/scripts/`](gsplat_generation/scripts). Needs the
`gsplat` submodule checked out (`git submodule update --init --recursive`)
and a CUDA GPU. No env vars — paths are passed as positional CLI args.
Typical flow: `vggt_infer.py` → `gsplat_full_train.py` (or
`gsplat_quality_train.py`) → `prune_gaussians.py` / `trim_gsplat_ply.py` →
`export_gsplat_files.py`.

### 3. WebXR viewer (browser-based playback)

```bash
cd webxr-viewer && python3 serve.py
```
Prints a local HTTPS URL (`https://<local-ip>:8443/`) — open it on a Quest
browser (or desktop) on the same network. Uses the checked-in self-signed
`cert.pem`/`key.pem` for local HTTPS. Used for a testing environment

### 4. Unity app (full VR experience)

Open `unity/` in the Unity Editor (Meta XR / OpenXR packages are already in
`Packages/manifest.json`), build for Quest, and point `AIInterface` at the
memory API server's local-network address (`http://<veriton-ip>:8020`,
started via `memory_bank/scripts/api_server.py`) in the relevant
Inspector field.

## Datasets & provenance

No synthetic or third-party datasets are used — every "memory" is a real
phone video recorded by the team for this demo, and the pipeline is
designed to work on arbitrary personal footage, not a fixed dataset.

- **Sample source videos** — `AkshaySampleVideos/` (`IMG_4745.MOV`,
  `IMG_4746.MOV`) and a few more under `gsplat_generation/`
  (`great_wall.mp4`, `IMG_8544.MOV`) — personal phone recordings used to
  exercise the pipeline end-to-end during development. These are the only
  "data" checked into the repo beyond code.
- **Pretrained model weights are not included in this repo** and are
  downloaded/deployed separately: VGGT (`facebook/VGGT-1B`, Meta, via
  Hugging Face) for pose/point-cloud inference, and the NVIDIA
  Nemotron/Parakeet family (Nano Omni, Embed, Lightning, Parakeet TDT) via
  NIM/Ollama on the Veriton. None of these are fine-tuned on the sample
  videos — they're used zero-shot/off-the-shelf.
- **Generated/derived artifacts** (trained `.ply`/`.pt`/`.splat` scenes,
  the FAISS index, caption/metadata JSON, extracted frames) are produced
  locally by running the pipeline on the source videos above. These are
  gitignored, not committed — see Data & privacy below.

## Data & privacy

Captured videos, trained scenes, and the memory bank's index/metadata are
gitignored — this is personal content and stays on-device. See
`.gitignore` for the full list (large `.ply`/`.pt`/`.splat` binaries,
`memory_bank/data/`, source videos under `gsplat_generation/`, and the
baked splat assets under `unity/Assets/Memories/`).

## Known limitations & next steps

- **No text-to-speech yet** — `/turn` returns `response_text` for the AI
  guide's reply, but the `/speak` endpoint (text → audio) isn't wired to a
  TTS backend yet; the client currently shows text without spoken audio.
- **Single global conversation, no multi-user/session support** —
  `api_server.py` keeps one in-memory chat history for the whole server.
  Fine for a single-headset demo, not for concurrent users.
- **Small-scale memory bank** — `faiss-cpu` with a flat index is used
  deliberately since the demo memory bank is only dozens of entries; it
  won't scale to a large personal archive without an ANN index and
  on-disk/DB-backed metadata instead of a single `memory_store.json`.
- **GPU memory contention on-device** — VLM, Embed, and ASR servers share
  one GPU; keeping all three resident required a manual NIM KV-cache
  workaround (`NIM_KV_CACHE_PERCENT=0.6`, see
  `memory_bank/memory_pipeline/gpu_swap.py`) rather than a documented flag.
- **aarch64-specific rough edges** — the Veriton's GB10 is aarch64 + CUDA
  13.0; some Python deps (`faiss-cpu`, `opencv-python`) need prebuilt
  aarch64 wheels checked, and gsplat/VGGT training scripts currently
  hardcode `/workspace/...`-style paths rather than exposing a CLI/config
  surface.
- **No automated tests / CI** — verification so far has been manual
  end-to-end runs on sample videos; unit/integration tests aren't in place.
- **Next steps:** wire up TTS end-to-end, make ingest auto-trigger gsplat
  training (or at least poll for scene completion), replace hardcoded
  script paths with proper CLI args/config, and add basic tests around the
  retrieval/resolution logic (match/ambiguous/no-match) since that's the
  crux of the voice UX.
