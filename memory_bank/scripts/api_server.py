#!/usr/bin/env python3
"""
HTTP API server for the Timebox voice pipeline — this is what Unity talks to.

Runs on the Veriton, reachable over the local network the same way the
WebXR viewer was (http://<veriton-ip>:8020). Unity sends either recorded
audio or (for testing without a mic) raw text, and gets back a structured
JSON response describing what happened — the cleaned query, search scores,
the resolution (match/ambiguous/no_match), the response text to speak, and
(on a match) the scene_path for the .splat file to load.

This deliberately does NOT do TTS itself — it returns text, and a separate
call to /speak (or the client doing it locally) turns that into audio. Kept
separate so Unity can choose when/whether to synthesize speech (e.g. it
might want to show the text immediately and start audio playback slightly
after, or skip audio entirely for a captions-only accessibility mode).

Endpoints:
  POST /turn        - the main endpoint: audio file OR raw text in, full
                       resolved turn out (see response shape below)
  POST /speak        - text in, audio (wav) out, via TTS
  GET  /health       - simple health check
  POST /chat/reset   - clear conversation history, start fresh

Run with:
    python3 scripts/api_server.py
    (or: uvicorn scripts.api_server:app --host 0.0.0.0 --port 8020)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from memory_pipeline import config, gpu_swap
from memory_pipeline.embedder import Embedder
from memory_pipeline.memory_store import MemoryStore
from memory_pipeline.query_engine import QueryEngine
from memory_pipeline.transcriber import Transcriber

# TTS client added once that piece is deployed — see the /speak endpoint
# below for exactly where it plugs in.
# from memory_pipeline.tts_client import Speaker

app = FastAPI(title="Timebox Voice API")

# Unity's HTTP client and the WebXR-style browser testing both need CORS
# open — this server only runs on the local network, so a permissive
# policy here doesn't meaningfully widen the actual attack surface.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models are constructed once at startup, not per-request — same objects
# your CLI scripts use, just held open for the life of the server instead
# of created fresh each invocation.
_engine = QueryEngine()
_embedder = Embedder()
_store = MemoryStore()
_transcriber = Transcriber()

# In-memory conversation history for the current session. A real multi-user
# deployment would key this per-session-id; for a single headset demo, one
# global conversation is the right amount of complexity.
_history: list[dict] = []


def _summarize_for_history(result, results) -> str:
    """Same logic as query_memory.py's version — kept in sync manually since
    this server intentionally doesn't import from scripts/ (scripts are
    meant to be run directly, not imported as a library)."""
    if result.status == "match":
        alt_names = [m.note or m.caption[:40] for m, _ in results[1:3]]
        alt_note = f" (other candidates: {', '.join(alt_names)})" if alt_names else ""
        return f"Loading memory: {result.memory.note or result.memory.caption[:60]}{alt_note}"
    elif result.status == "ambiguous":
        names = [m.note or m.caption[:40] for m in result.candidates]
        return f"Multiple matches found: {', '.join(names)}. {result.message}"
    else:
        return result.message


class TurnResponse(BaseModel):
    heard: str  # what the user said (post-transcription if audio was sent)
    cleaned_query: str
    status: str  # "match" | "ambiguous" | "no_match"
    response_text: str  # what the assistant should say/display back
    scores: list[dict]  # [{"note": ..., "score": ...}, ...] top candidates
    memory_id: str | None = None
    scene_path: str | None = None
    candidates: list[dict] | None = None  # populated when status == "ambiguous"


@app.get("/health")
def health():
    return {"status": "ok", "memories_in_store": len(_store)}


@app.post("/chat/reset")
def reset_chat():
    global _history
    _history = []
    return {"status": "reset"}


@app.post("/turn", response_model=TurnResponse)
async def turn(
    audio: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
):
    """
    Run one full voice-pipeline turn. Send EITHER an audio file (field name
    "audio") OR raw text (field name "text") — not both. Audio goes through
    Parakeet ASR first; text skips straight to the reasoning step (useful
    for testing without a microphone, or for a text-fallback input mode).
    """
    if audio is None and not text:
        raise HTTPException(400, "Provide either an 'audio' file or a 'text' field.")

    if len(_store) == 0:
        raise HTTPException(409, "Memory bank is empty — ingest a memory first.")

    # --- Get the raw query text, either from ASR or directly ---
    if audio is not None:
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        try:
            raw_query = _transcriber.transcribe(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        raw_query = text.strip()

    if not raw_query:
        raise HTTPException(422, "Got empty transcription/text — nothing to search for.")

    # --- The existing pipeline, unchanged ---
    clean = _engine.clean_query(raw_query, conversation_history=_history)

    gpu_swap.ensure_embed_active()
    query_vec = _embedder.embed_query(clean)
    results = _store.search(query_vec)

    result = _engine.resolve_results(raw_query, results)

    # --- Update conversation history for the next turn ---
    _history.append({"role": "user", "content": raw_query})
    _history.append({"role": "assistant", "content": _summarize_for_history(result, results)})

    # --- Shape the response for Unity ---
    response = TurnResponse(
        heard=raw_query,
        cleaned_query=clean,
        status=result.status,
        response_text=result.message,
        scores=[{"note": m.note or m.caption[:60], "score": round(s, 4)} for m, s in results],
    )

    if result.status == "match":
        response.memory_id = result.memory.id
        response.scene_path = result.memory.scene_path
    elif result.status == "ambiguous":
        response.candidates = [
            {"id": m.id, "note": m.note or m.caption[:60]} for m in result.candidates
        ]

    return response


@app.post("/speak")
async def speak(text: str = Form(...)):
    """
    Text -> speech audio (wav bytes). Placeholder until the TTS piece is
    deployed — returns 501 for now so Unity gets a clear, explicit signal
    that audio isn't available yet rather than a silent failure or a
    misleading empty 200 response.
    """
    raise HTTPException(
        501,
        "TTS not yet deployed — /turn already returns response_text; "
        "Unity can display that as captions until /speak is implemented.",
    )
    # Once Magpie TTS (or similar) is deployed, this becomes:
    #   audio_bytes = _speaker.synthesize(text)
    #   return Response(content=audio_bytes, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8020)
