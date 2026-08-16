"""
The memory bank itself: a FAISS vector index for similarity search, plus a
JSON sidecar file holding each memory's metadata (note, caption, video path,
3D scene pointer, timestamp).

FAISS only stores vectors + an integer ID — it knows nothing about your data.
We keep a parallel JSON store keyed by that same integer ID so a vector match
can be turned back into an actual memory record.

This is intentionally simple (flat index, exact search) rather than an
approximate/production index — with a hackathon-scale number of memories
(dozens, not millions), exact search is fast enough and much easier to reason
about than tuning an ANN index under time pressure.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import faiss
import numpy as np

from . import config


@dataclass
class Memory:
    id: str
    note: str
    caption: str
    video_path: str
    scene_path: str | None = None  # pointer to the trained 3D Gaussian splat scene
    enrichment: str = ""  # Lightning-generated alternative phrasings, widens retrieval surface area
    created_at: float = field(default_factory=time.time)

    @property
    def searchable_text(self) -> str:
        """What actually gets embedded: note + caption + enrichment combined."""
        parts = [p for p in [self.note, self.caption, self.enrichment] if p]
        return " | ".join(parts)


class MemoryStore:
    def __init__(
        self,
        index_path: Path = config.FAISS_INDEX_PATH,
        metadata_path: Path = config.MEMORY_STORE_PATH,
        dim: int = config.EMBEDDING_DIM,
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.dim = dim

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # id_map: FAISS internal row index -> Memory.id, since FAISS itself
        # only deals in contiguous integer positions, not our string ids.
        self.id_map: list[str] = []
        self.memories: dict[str, Memory] = {}

        if self.index_path.exists() and self.metadata_path.exists():
            self._load()
        else:
            # Inner product on normalized vectors == cosine similarity.
            self.index = faiss.IndexFlatIP(self.dim)

    def _load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "r") as f:
            raw = json.load(f)
        self.id_map = raw["id_map"]
        self.memories = {mid: Memory(**m) for mid, m in raw["memories"].items()}

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump(
                {
                    "id_map": self.id_map,
                    "memories": {mid: asdict(m) for mid, m in self.memories.items()},
                },
                f,
                indent=2,
            )

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def add_memory(
        self,
        note: str,
        caption: str,
        video_path: str,
        embedding: np.ndarray,
        scene_path: str | None = None,
        enrichment: str = "",
    ) -> Memory:
        """Add a new memory to the bank. `embedding` should be the vector for
        the memory's searchable_text (note + caption + enrichment), produced
        by Embedder.embed_document()."""
        memory = Memory(
            id=str(uuid.uuid4()),
            note=note,
            caption=caption,
            video_path=str(video_path),
            scene_path=scene_path,
            enrichment=enrichment,
        )

        vec = self._normalize(embedding).reshape(1, -1).astype(np.float32)
        self.index.add(vec)
        self.id_map.append(memory.id)
        self.memories[memory.id] = memory

        self._save()
        return memory

    def search(self, query_embedding: np.ndarray, top_k: int = config.TOP_K_RESULTS):
        """Return up to top_k (Memory, score) pairs, best match first."""
        if self.index.ntotal == 0:
            return []

        vec = self._normalize(query_embedding).reshape(1, -1).astype(np.float32)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            memory_id = self.id_map[idx]
            results.append((self.memories[memory_id], float(score)))

        return results

    def get_memory(self, memory_id: str) -> Memory:
        """Look up a memory by its id. Raises KeyError if it doesn't exist —
        callers should catch this or check `memory_id in store.memories`
        first if a missing id is a normal/expected case."""
        return self.memories[memory_id]

    def set_scene_path(self, memory_id: str, scene_path: str) -> Memory:
        """
        Attach (or update) a memory's .splat scene path after the fact.

        This exists because ingestion (VLM captioning + embedding) and 3D
        reconstruction (gsplat training) are independent, often-parallel
        pipelines that don't have to finish in lockstep — you can ingest a
        memory as soon as its source video/photos exist, search for it
        immediately, and attach the real scene_path once reconstruction
        catches up, without re-running captioning or re-embedding anything.

        Does NOT touch the embedding/vector — only the metadata changes,
        so this is a cheap update, not a re-ingest.
        """
        if memory_id not in self.memories:
            raise KeyError(f"No memory with id {memory_id!r} in the store.")
        self.memories[memory_id].scene_path = scene_path
        self._save()
        return self.memories[memory_id]

    def __len__(self) -> int:
        return len(self.memories)