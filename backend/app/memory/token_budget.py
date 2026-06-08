"""
app/memory/token_budget.py
───────────────────────────
Token budget manager for context window management.

Computes available token budget for retrieved context, accounting for:
  - system prompt
  - agent instructions
  - conversation history
  - user query
  - reserved output tokens

Uses the same HuggingFace tokenizer as the chunker for consistency.
All token counts are exact (not estimated by character heuristics).

Future agents call this before building their prompts to know
exactly how many context tokens are available.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import Settings
from app.core.exceptions import TokenBudgetException

logger = logging.getLogger(__name__)


class TokenBudgetManager:
    """
    Manages the LLM context window token budget.

    All token counts use the same AutoTokenizer as the chunking pipeline,
    ensuring that chunk token_count values and budget calculations are
    consistent.

    Args:
        settings:   Application settings (reads MAX_CONTEXT_TOKENS,
                    RESERVED_OUTPUT_TOKENS, TOKENIZER_MODEL).
        tokenizer:  Optional pre-loaded tokenizer instance. If not provided,
                    a new one is loaded lazily on first use.
    """

    def __init__(
        self,
        settings: Settings,
        tokenizer: Any | None = None,
    ) -> None:
        self._settings = settings
        self._tokenizer = tokenizer  # injected or lazy-loaded

    # ── Tokenizer ─────────────────────────────────────────────────────────────

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer

        model_name = self._settings.tokenizer_model
        try:
            from transformers import AutoTokenizer  # type: ignore[import-untyped]

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True
            )
            logger.debug(
                "TokenBudgetManager: tokenizer loaded",
                extra={"model": model_name},
            )
        except Exception as exc:
            raise TokenBudgetException(
                f"Failed to load tokenizer '{model_name}' for budget calculation: {exc}",
                context={"tokenizer_model": model_name},
            ) from exc

        return self._tokenizer

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate_tokens(self, text: str) -> int:
        """
        Return the exact token count for ``text`` using the configured tokenizer.

        Args:
            text: Input string.

        Returns:
            Token count (integer).
        """
        if not text:
            return 0
        tokenizer = self._get_tokenizer()
        ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
        return len(ids)

    def calculate_available_context(
        self,
        system_prompt: str = "",
        instructions: str = "",
        history: str = "",
        query: str = "",
        reserved_output: int | None = None,
    ) -> int:
        """
        Calculate how many tokens remain for retrieved context.

        Budget formula:
            available = max_context - system - instructions - history
                        - query - reserved_output

        Args:
            system_prompt:   System prompt text.
            instructions:    Agent instructions text.
            history:         Conversation history text.
            query:           Current user query.
            reserved_output: Override for reserved output token count.

        Returns:
            Available token budget for context injection.

        Raises:
            TokenBudgetException: If the fixed components exceed the context window.
        """
        reserved = reserved_output if reserved_output is not None else self._settings.reserved_output_tokens
        max_tokens = self._settings.max_context_tokens

        system_tokens = self.estimate_tokens(system_prompt)
        instruction_tokens = self.estimate_tokens(instructions)
        history_tokens = self.estimate_tokens(history)
        query_tokens = self.estimate_tokens(query)

        fixed_total = system_tokens + instruction_tokens + history_tokens + query_tokens + reserved
        available = max_tokens - fixed_total

        logger.debug(
            "Token budget calculated",
            extra={
                "max_context": max_tokens,
                "system": system_tokens,
                "instructions": instruction_tokens,
                "history": history_tokens,
                "query": query_tokens,
                "reserved_output": reserved,
                "available_for_context": available,
            },
        )

        if available <= 0:
            raise TokenBudgetException(
                f"Fixed prompt components ({fixed_total} tokens) exceed the "
                f"context window ({max_tokens} tokens). No budget for context.",
                context={
                    "max_context": max_tokens,
                    "fixed_total": fixed_total,
                    "available": available,
                },
            )

        return available

    def allocate_chunk_budget(
        self,
        available_tokens: int,
        chunks: list[Any],
    ) -> list[Any]:
        """
        Select as many chunks as fit within ``available_tokens``.

        Chunks are taken in order (highest rerank_score first is expected
        from the caller). The selection stops when the next chunk would
        exceed the budget.

        Args:
            available_tokens: Token budget for chunk content.
            chunks:           Ordered list of :class:`RetrievedChunk` or
                              :class:`DocumentChunk` instances with a
                              ``token_count`` attribute.

        Returns:
            Subset of chunks that fit within the budget.
        """
        selected: list[Any] = []
        remaining = available_tokens

        for chunk in chunks:
            if chunk.token_count <= remaining:
                selected.append(chunk)
                remaining -= chunk.token_count
            else:
                # Try to fit a truncated version? No — we keep chunks whole
                # to preserve semantic integrity.
                logger.debug(
                    "Chunk exceeds remaining budget — stopping allocation",
                    extra={
                        "chunk_tokens": chunk.token_count,
                        "remaining": remaining,
                    },
                )
                break

        logger.info(
            "Chunk budget allocated",
            extra={
                "requested": len(chunks),
                "selected": len(selected),
                "tokens_used": available_tokens - remaining,
                "tokens_available": available_tokens,
            },
        )

        return selected

    def truncate_context(self, text: str, max_tokens: int) -> str:
        """
        Truncate ``text`` to at most ``max_tokens`` tokens.

        Decodes the truncated token sequence back to text.
        Used as a last-resort safety net.

        Args:
            text:       Input text to truncate.
            max_tokens: Maximum number of tokens to retain.

        Returns:
            Truncated text string.
        """
        if not text:
            return text

        tokenizer = self._get_tokenizer()
        ids: list[int] = tokenizer.encode(text, add_special_tokens=False)

        if len(ids) <= max_tokens:
            return text

        truncated_ids = ids[:max_tokens]
        result: str = tokenizer.decode(
            truncated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        logger.debug(
            "Context truncated",
            extra={
                "original_tokens": len(ids),
                "truncated_to": max_tokens,
            },
        )

        return result
