"""
Speech-to-text via the local Parakeet ASR server.

Parakeet exposes an OpenAI-compatible /v1/audio/transcriptions endpoint, so
this uses the same `openai` client library as every other model in this
pipeline — just calling .audio.transcriptions.create() instead of
.chat.completions.create() or .embeddings.create().

This is the entry point to the voice pipeline: audio in, text out. That text
becomes the raw_query fed into QueryEngine.clean_query(), same as if the user
had typed it into query_memory.py — nothing downstream needs to know the
query originated as speech rather than text.
"""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from . import config


class Transcriber:
    def __init__(self, base_url: str = config.ASR_BASE_URL, model: str = config.ASR_MODEL_NAME):
        self.client = OpenAI(base_url=base_url, api_key=config.LOCAL_API_KEY)
        self.model = model

    def transcribe(self, audio_path: str | Path, language: str = "en") -> str:
        """
        Transcribe an audio file to text. Accepts wav, mp3, flac, ogg, webm,
        m4a, mp4 per Parakeet's supported formats.

        `language` defaults to "en" rather than "auto" — auto-detection adds
        a small amount of latency and, for a single-user demo where the
        language is known ahead of time, isn't worth the tradeoff. Pass
        language="auto" explicitly if you need detection.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
                language=language,
                response_format="json",
            )

        # response is a Transcription object with a .text field, per the
        # OpenAI-compatible schema.
        text = response.text.strip()
        if not text:
            raise ValueError(
                "Transcription came back empty — check the audio file has "
                "actual speech and isn't silent/corrupted."
            )
        return text
