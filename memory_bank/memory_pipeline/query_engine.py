"""
Stage 3: the reasoning layer, backed by the local Nemotron 3.5 Lightning
server.

This does the two jobs that a raw vector search can't do on its own:

  1. Turn a conversational user request ("I want to revisit my 18th
     birthday", or a follow-up like "no, the other one") into a clean
     search query worth embedding.
  2. Look at the vector search results and decide what to do — confidently
     return the top match, or recognize two results are close enough that
     the user should be asked to disambiguate.

Nothing here does similarity math — that's MemoryStore.search(). Lightning's
only job is language: cleaning up input, and turning ranked results into a
decision/response.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from . import config
from .memory_store import Memory


@dataclass
class QueryResult:
    status: str  # "match" | "ambiguous" | "no_match"
    message: str
    memory: Memory | None = None
    candidates: list[Memory] | None = None


class QueryEngine:
    def __init__(
        self, base_url: str = config.LIGHTNING_BASE_URL, model: str = config.LIGHTNING_MODEL_NAME
    ):
        self.client = OpenAI(base_url=base_url, api_key=config.LOCAL_API_KEY)
        self.model = model

    def clean_query(self, raw_query: str, conversation_history: list[dict] | None = None) -> str:
        """
        Turn a raw, possibly conversational user request into a clean search
        string. conversation_history (list of {"role", "content"} dicts) lets
        this resolve follow-ups like "no, the other one" using prior turns.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite a user's request into a short, clean search "
                    "query for a personal memory database. Strip filler words "
                    "and hedging, resolve pronouns/references using the "
                    "conversation history if given, and output ONLY the "
                    "search query text — no explanation, no quotes."
                ),
            }
        ]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": raw_query})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=64,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def resolve_results(
        self, raw_query: str, results: list[tuple[Memory, float]]
    ) -> QueryResult:
        """
        Decide what to do with ranked (Memory, score) results:
          - no results at all -> no_match
          - a clear winner -> match
          - top two scores too close together -> ambiguous, ask the user
        """
        if not results:
            return QueryResult(
                status="no_match",
                message="I couldn't find a memory matching that. Could you describe it differently?",
            )

        top_memory, top_score = results[0]

        if len(results) == 1 or (top_score - results[1][1]) > config.AMBIGUOUS_SCORE_GAP:
            return QueryResult(
                status="match",
                message=f"Loading memory: {top_memory.note or top_memory.caption[:60]}",
                memory=top_memory,
            )

        # Ambiguous: let Lightning phrase a natural disambiguation question
        candidates = [m for m, _ in results[:3]]
        candidate_descriptions = "\n".join(
            f"- {m.note or m.caption[:80]}" for m in candidates
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "The user asked for a memory and multiple close matches "
                        "were found. Write one short, friendly sentence asking "
                        "which one they meant, referencing the specific options "
                        "given."
                    ),
                },
                {
                    "role": "user",
                    "content": f"User asked: '{raw_query}'\n\nCandidate memories:\n{candidate_descriptions}",
                },
            ],
            max_tokens=100,
            temperature=0.4,
        )

        return QueryResult(
            status="ambiguous",
            message=response.choices[0].message.content.strip(),
            candidates=candidates,
        )
