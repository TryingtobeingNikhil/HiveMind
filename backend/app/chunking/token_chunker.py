"""
app/chunking/token_chunker.py
──────────────────────────────
Token-aware document chunker using HuggingFace AutoTokenizer.

Design:
- Encodes the full document to token IDs.
- Slides a window of `chunk_token_size` with `overlap_tokens` overlap.
- Decodes each window back to text to create a DocumentChunk.
- Output is strictly deterministic for a given input + settings combination.

EXECUTION MODEL:
- Chunking is a pure CPU operation (no I/O, no LLM calls).
- The tokenizer is loaded once and cached for the application lifetime.
- Sequential only — no concurrent chunking pipelines.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from app.core.config import Settings
from app.core.exceptions import DocumentParsingException
from app.schemas.documents import DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


class TokenAwareChunker:
    """
    Splits a document into fixed-size, overlapping token chunks.

    The tokenizer is loaded lazily on first use and cached.
    All token counts are exact (verified against the tokenizer output).

    Args:
        settings: Application settings (reads TOKENIZER_MODEL, CHUNK_TOKEN_SIZE,
                  CHUNK_OVERLAP_TOKENS).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tokenizer: Any | None = None
        self._lock = threading.Lock()  # protect lazy init in threaded contexts

    # ── Tokenizer lifecycle ───────────────────────────────────────────────────

    def _get_tokenizer(self) -> Any:
        """
        Lazily load and cache the HuggingFace tokenizer.

        Thread-safe via a lock. The tokenizer is downloaded to
        ~/.cache/huggingface/ on first use and reused from cache thereafter.
        """
        if self._tokenizer is not None:
            return self._tokenizer

        with self._lock:
            if self._tokenizer is not None:  # double-checked locking
                return self._tokenizer

            model_name = self._settings.tokenizer_model
            logger.info(
                "Loading tokenizer",
                extra={"model": model_name},
            )
            try:
                from transformers import AutoTokenizer  # type: ignore[import-untyped]

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=True,  # Use Rust-backed fast tokenizer when available
                )
                self._tokenizer = tokenizer
                logger.info(
                    "Tokenizer loaded",
                    extra={"model": model_name, "vocab_size": tokenizer.vocab_size},
                )
            except Exception as exc:
                raise DocumentParsingException(
                    f"Failed to load tokenizer '{model_name}': {exc}. "
                    f"Ensure the model name is correct and network/cache is accessible.",
                    context={"tokenizer_model": model_name},
                ) from exc

        return self._tokenizer

    # ── Public API ────────────────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """
        Return the exact token count for a string using the configured tokenizer.

        Args:
            text: Input text.

        Returns:
            Number of tokens (excluding special tokens).
        """
        tokenizer = self._get_tokenizer()
        ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
        return len(ids)

    def chunk(
        self,
        content: str,
        metadata: DocumentMetadata,
    ) -> list[DocumentChunk]:
        """
        Split ``content`` into token-aware overlapping chunks.

        Args:
            content:  Full extracted document text.
            metadata: Source document metadata (for IDs and attribution).

        Returns:
            Ordered list of :class:`DocumentChunk` instances.
            Each chunk has an exact ``token_count`` verified by the tokenizer.

        Raises:
            DocumentParsingException: If the content is empty or tokeniser fails.
        """
        if not content.strip():
            raise DocumentParsingException(
                "Cannot chunk empty document content.",
                context={"document_id": metadata.document_id},
            )

        chunk_size = self._settings.chunk_token_size
        overlap = self._settings.chunk_overlap_tokens
        stride = chunk_size - overlap  # tokens advanced per chunk

        tokenizer = self._get_tokenizer()

        logger.info(
            "Chunking document",
            extra={
                "document_id": metadata.document_id,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "content_len": len(content),
            },
        )

        try:
            token_ids: list[int] = tokenizer.encode(
                content, add_special_tokens=False
            )
        except Exception as exc:
            raise DocumentParsingException(
                f"Tokenization failed for document '{metadata.document_id}': {exc}",
                context={"document_id": metadata.document_id},
            ) from exc

        total_tokens = len(token_ids)

        if total_tokens == 0:
            raise DocumentParsingException(
                "Document produced zero tokens after tokenization.",
                context={"document_id": metadata.document_id},
            )

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_index = 0

        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_token_ids = token_ids[start:end]

            # Decode back to text — may differ slightly from original due to
            # tokenizer normalisation, but content is semantically equivalent.
            try:
                chunk_text = tokenizer.decode(
                    chunk_token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            except Exception as exc:
                raise DocumentParsingException(
                    f"Failed to decode chunk {chunk_index}: {exc}",
                    context={"document_id": metadata.document_id, "chunk_index": chunk_index},
                ) from exc

            chunk_text = chunk_text.strip()
            if chunk_text:  # skip empty decoded chunks (e.g. pure whitespace tokens)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{metadata.document_id}_chunk_{chunk_index}",
                        document_id=metadata.document_id,
                        chunk_index=chunk_index,
                        token_count=len(chunk_token_ids),
                        content=chunk_text,
                        filename=metadata.filename,
                    )
                )
                chunk_index += 1

            if end == total_tokens:
                break  # reached end of document

            start += stride

        logger.info(
            "Chunking complete",
            extra={
                "document_id": metadata.document_id,
                "total_tokens": total_tokens,
                "chunk_count": len(chunks),
            },
        )

        return chunks
