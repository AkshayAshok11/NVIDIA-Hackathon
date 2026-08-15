"""
Stage 1 of the memory pipeline: VLM captioning.

Takes sampled frames from a video and asks the local VLM to describe what's
in the scene. This produces the rich, auto-generated text signal that gets
embedded alongside the user's own note — the whole point being that a thin
user note ("bday") still gets matched later because the VLM's caption is
descriptive ("birthday cake with lit candles, balloons, three people").

Talks to whatever local server is running at config.VLM_BASE_URL. Point this
at your NIM/Ollama/whatever-you-deploy endpoint before running.
"""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from . import config
from .video_utils import image_to_base64


class VLMCaptioner:
    def __init__(self, base_url: str = config.VLM_BASE_URL, model: str = config.VLM_MODEL_NAME):
        self.client = OpenAI(base_url=base_url, api_key=config.LOCAL_API_KEY)
        self.model = model

    def caption_frames(self, frame_paths: list[Path], user_note: str = "") -> str:
        """
        Send frames to the VLM and get back a single descriptive caption
        covering the whole clip. If the user supplied a note, it's passed
        as context so the VLM's caption can complement rather than repeat it.
        """
        if not frame_paths:
            raise ValueError("No frames provided to caption.")

        content = [
            {
                "type": "text",
                "text": (
                    "You are describing a personal memory captured on video for a "
                    "memory-archival app. Look at these frames from a single clip "
                    "and write one descriptive paragraph covering: the setting/room, "
                    "notable objects, people present (described generically, not "
                    "identified by name), and any activity or occasion visible "
                    "(e.g. a celebration, a quiet room, decorations). Be concrete "
                    "and specific — this text will be used to search for this "
                    "memory later, so include distinguishing details."
                    + (f"\n\nThe user's own note for this memory: '{user_note}'" if user_note else "")
                ),
            }
        ]

        for frame_path in frame_paths:
            b64 = image_to_base64(frame_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1500,
            temperature=0.3,
        )

        content = response.choices[0].message.content
        if content is None:
            # Model ran out of tokens before finishing — fall back to the
            # reasoning trace if available, rather than crashing. If you see
            # this path taken often, max_tokens is still too tight.
            content = getattr(response.choices[0].message, "reasoning", "") or ""
        return content.strip()