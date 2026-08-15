"""
Stage 2 of the memory pipeline: embeddings.

Turns text (VLM captions, user notes, and later, user queries) into vectors
via the local Nemotron Embed server. This is pure text-in/vector-out — no
reasoning, no comparison logic. See embedder.py's callers (memory_store.py,
query_engine.py) for what happens with the vectors afterward.

Nemotron embedding models distinguish between "document" text (things being
indexed) and "query" text (the thing you're searching with) — passing the
right input_type improves retrieval quality, so we expose both explicitly
rather than a single generic embed() call.
"""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from . import config


class Embedder:
    def __init__(self, base_url: str = config.EMBED_BASE_URL, model: str = config.EMBED_MODEL_NAME):
        self.client = OpenAI(base_url=base_url, api_key=config.LOCAL_API_KEY)
        self.model = model

    def _embed(self, text: str, input_type: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            extra_body={"input_type": input_type},  # "query" or "passage"/"document"
        )
        vector = response.data[0].embedding
        return np.array(vector, dtype=np.float32)

    def embed_document(self, text: str) -> np.ndarray:
        """Embed text that's being stored/indexed (a memory's caption + note)."""
        return self._embed(text, input_type="passage")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a user's search query."""
        return self._embed(text, input_type="query")
