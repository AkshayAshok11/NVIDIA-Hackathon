# Timebox

Timebox turns a short phone video of a moment into a place you can walk back
into. Record a memory, and Timebox reconstructs it as a navigable 3D
Gaussian-splat scene, indexes it against a voice-searchable memory bank, and
lets you revisit it in VR — just ask for it out loud and step through a
portal into the reconstructed scene, with an AI guide alongside you.

Built for an NVIDIA hackathon, running end-to-end **locally on a DGX Spark
(Acer Veriton GN100)** — no cloud calls, so memory content never leaves the
device.

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

## Setup

Each pipeline component has its own environment; see the linked docs for
full detail.

1. **Memory pipeline** (captioning, embedding, retrieval, voice API server)
   — [`memory_bank/README.md`](memory_bank/README.md). Requires three local
   OpenAI-compatible model servers (VLM, Embed, Lightning) plus an ASR
   server already running on the Veriton.
2. **Gsplat generation** (video → 3D scene) — see
   [`gsplat_generation/scripts/`](gsplat_generation/scripts). Needs the
   `gsplat` submodule checked out (`git submodule update --init`) and a CUDA
   GPU. Typical flow: `vggt_infer.py` → `gsplat_full_train.py` (or
   `gsplat_quality_train.py`) → `prune_gaussians.py` /
   `trim_gsplat_ply.py` → `export_gsplat_files.py`.
3. **WebXR viewer** (browser-based playback) —
   `cd webxr-viewer && python3 serve.py`, then open the printed HTTPS URL
   on a Quest browser on the same network.
4. **Unity app** (full VR experience) — open `unity/` in the Unity Editor
   (Meta XR / OpenXR packages are already in `Packages/manifest.json`),
   build for Quest, and point `AIInterface` at the memory API server's
   local-network address (see `memory_bank/scripts/api_server.py`).

## Data & privacy

Captured videos, trained scenes, and the memory bank's index/metadata are
gitignored — this is personal content and stays on-device. See
`.gitignore` for the full list (large `.ply`/`.pt`/`.splat` binaries,
`memory_bank/data/`, source videos under `gsplat_generation/`, and the
baked splat assets under `unity/Assets/Memories/`).
