"""
app/memory/context_builder.py
──────────────────────────────
Context builder — assembles the final ContextPackage for LLM injection.

Responsibilities:
  1. Deduplicate retrieved chunks (by content hash)
  2. Sort chunks deterministically (by rerank_score desc, chunk_index asc)
  3. Apply token budget constraints
  4. Format chunks into a structured context string
  5. Preserve full source attribution for every included chunk
  6. Collect token usage statistics

Output (ContextPackage) is consumed directly by future agents.
"""

from __future__ import annotations

import hashlib
import logging

from app.core.config import Settings
from app.memory.token_budget import TokenBudgetManager
from app.schemas.retrieval import (
    ContextPackage,
    ContextSource,
    RetrievedChunk,
    TokenUsage,
)

logger = logging.getLogger(__name__)

_CHUNK_SEPARATOR = "\n\n---\n\n"
_CHUNK_TEMPLATE = "[Source: {filename} | Chunk {chunk_index}]\n{content}"


class ContextBuilder:
    """
    Builds a :class:`ContextPackage` from a list of retrieved chunks.

    Args:
        settings:       Application settings.
        budget_manager: Shared :class:`TokenBudgetManager` instance.
    """

    def __init__(
        self,
        settings: Settings,
        budget_manager: TokenBudgetManager,
    ) -> None:
        self._settings = settings
        self._budget = budget_manager

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        system_prompt: str = "",
        instructions: str = "",
        history: str = "",
        reserved_output: int | None = None,
    ) -> ContextPackage:
        """
        Build a ContextPackage from retrieved chunks.

        Pipeline:
            deduplicate → sort → budget allocation → format → package

        Args:
            query:          Original search query.
            chunks:         Retrieved (and optionally reranked) chunks.
            system_prompt:  System prompt text (for budget calculation).
            instructions:   Agent instructions (for budget calculation).
            history:        Conversation history (for budget calculation).
            reserved_output: Override reserved output token count.

        Returns:
            A fully assembled :class:`ContextPackage`.
        """
        if not chunks:
            return self._empty_package(query)

        logger.info(
            "Building context package",
            extra={"query_len": len(query), "input_chunks": len(chunks)},
        )

        # 1. Deduplicate by content hash
        unique_chunks, dedup_count = self._deduplicate(chunks)

        # 2. Sort: rerank_score desc → retrieval_score desc → chunk_index asc
        sorted_chunks = self._sort_chunks(unique_chunks)

        # 3. Calculate available token budget
        try:
            available_tokens = self._budget.calculate_available_context(
                system_prompt=system_prompt,
                instructions=instructions,
                history=history,
                query=query,
                reserved_output=reserved_output,
            )
        except Exception as exc:
            logger.warning(
                "Token budget calculation failed — returning empty context",
                extra={"error": str(exc)},
            )
            return self._empty_package(query)

        # 4. Allocate chunks within budget
        selected_chunks = self._budget.allocate_chunk_budget(
            available_tokens, sorted_chunks
        )

        if not selected_chunks:
            logger.warning(
                "No chunks fit within token budget",
                extra={"available": available_tokens},
            )
            return self._empty_package(query)

        # 5. Format context string
        formatted_context = self._format_chunks(selected_chunks)

        # 6. Build source attribution
        sources = [
            ContextSource(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                retrieval_score=c.retrieval_score,
                rerank_score=c.rerank_score,
            )
            for c in selected_chunks
        ]

        # 7. Collect token usage statistics
        context_tokens = self._budget.estimate_tokens(formatted_context)
        system_tokens = self._budget.estimate_tokens(system_prompt)
        instruction_tokens = self._budget.estimate_tokens(instructions)
        history_tokens = self._budget.estimate_tokens(history)
        query_tokens = self._budget.estimate_tokens(query)
        reserved = reserved_output if reserved_output is not None else self._settings.reserved_output_tokens

        total_used = system_tokens + instruction_tokens + history_tokens + query_tokens + context_tokens + reserved

        token_usage = TokenUsage(
            system_prompt=system_tokens,
            instructions=instruction_tokens,
            history=history_tokens,
            query=query_tokens,
            retrieved_context=context_tokens,
            reserved_output=reserved,
            total_used=total_used,
            budget_available=self._settings.max_context_tokens,
            budget_remaining=max(0, self._settings.max_context_tokens - total_used),
        )

        logger.info(
            "Context package built",
            extra={
                "chunks_selected": len(selected_chunks),
                "chunks_deduped": dedup_count,
                "context_tokens": context_tokens,
                "total_tokens_used": total_used,
                "budget_remaining": token_usage.budget_remaining,
            },
        )

        return ContextPackage(
            query=query,
            formatted_context=formatted_context,
            sources=sources,
            token_usage=token_usage,
            chunk_count=len(selected_chunks),
            deduplication_count=dedup_count,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _content_hash(text: str) -> str:
        """Return a short MD5 hash of the chunk content for deduplication."""
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:12]

    def _deduplicate(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], int]:
        """
        Remove duplicate chunks by content hash.

        When duplicates exist, keep the one with the higher score.
        """
        seen: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            key = self._content_hash(chunk.content)
            existing = seen.get(key)
            if existing is None:
                seen[key] = chunk
            else:
                # Keep the higher-scored chunk
                current_score = chunk.rerank_score or chunk.retrieval_score
                existing_score = existing.rerank_score or existing.retrieval_score
                if current_score > existing_score:
                    seen[key] = chunk

        unique = list(seen.values())
        dedup_count = len(chunks) - len(unique)

        if dedup_count > 0:
            logger.debug(
                "Deduplicated chunks",
                extra={"removed": dedup_count, "remaining": len(unique)},
            )

        return unique, dedup_count

    @staticmethod
    def _sort_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Sort chunks deterministically.

        Order: rerank_score desc → retrieval_score desc → chunk_index asc

        Using chunk_index as a tiebreaker ensures output is deterministic
        even when scores are equal.
        """
        return sorted(
            chunks,
            key=lambda c: (
                -(c.rerank_score if c.rerank_score is not None else c.retrieval_score),
                -c.retrieval_score,
                c.chunk_index,
            ),
        )

    @staticmethod
    def _format_chunks(chunks: list[RetrievedChunk]) -> str:
        """
        Format chunks into a structured context string for LLM injection.

        Each chunk is labelled with its source filename and chunk index
        so the LLM can attribute answers to specific sources.
        """
        parts = [
            _CHUNK_TEMPLATE.format(
                filename=c.filename or "unknown",
                chunk_index=c.chunk_index,
                content=c.content.strip(),
            )
            for c in chunks
        ]
        return _CHUNK_SEPARATOR.join(parts)

    @staticmethod
    def _empty_package(query: str) -> ContextPackage:
        return ContextPackage(
            query=query,
            formatted_context="",
            sources=[],
            token_usage=TokenUsage(),
            chunk_count=0,
            deduplication_count=0,
        )
