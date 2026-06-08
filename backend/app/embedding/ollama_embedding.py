"""
app/embedding/ollama_embedding.py
──────────────────────────────────
Ollama-backed embedding provider.

Calls the Ollama /api/embeddings REST endpoint.
Requires the embedding model to be pulled locally:
    ollama pull nomic-embed-text

EXECUTION MODEL:
- embed_batch() loops sequentially — ONE HTTP call at a time.
- No concurrent requests to Ollama.
- Compatible with single-active-execution constraint.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.core.config import Settings
from app.core.exceptions import EmbeddingException, EmbeddingModelNotReadyException
from app.embedding.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider backed by Ollama's /api/embeddings endpoint.

    Args:
        settings: Application settings.
        http_client: Optional injected httpx client (for testing).
    """

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._base_url = settings.ollama_api_url
        self._model = settings.embedding_model
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=settings.embedding_timeout,
            write=30.0,
            pool=5.0,
        )
        self._client: httpx.AsyncClient = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Content-Type": "application/json"},
        )
        logger.debug(
            "OllamaEmbeddingProvider initialised",
            extra={"base_url": self._base_url, "model": self._model},
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaEmbeddingProvider":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # ── Public API ────────────────────────────────────────────────────────────

    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string via Ollama.

        Args:
            text: Input text (should be a single chunk or query).

        Returns:
            Float embedding vector.
        """
        if not text.strip():
            raise EmbeddingException(
                "Cannot embed empty text.",
                context={"model": self._model},
            )

        logger.debug(
            "Embedding text",
            extra={"model": self._model, "text_len": len(text)},
        )

        return await self._call_embeddings_api(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts sequentially (one at a time).

        Sequential per the single-active-execution constraint.
        """
        if not texts:
            return []

        logger.info(
            "Embedding batch",
            extra={"model": self._model, "batch_size": len(texts)},
        )

        embeddings: list[list[float]] = []
        for i, text in enumerate(texts):
            if not text.strip():
                logger.warning(
                    "Skipping empty text in batch",
                    extra={"index": i, "model": self._model},
                )
                # Use a zero vector as placeholder for empty texts
                embeddings.append([])
                continue
            embedding = await self._call_embeddings_api(text)
            embeddings.append(embedding)

        logger.info(
            "Batch embedding complete",
            extra={"model": self._model, "count": len(embeddings)},
        )

        return embeddings

    async def health_check(self) -> bool:
        """
        Verify the embedding model is available in Ollama.

        Returns:
            True if the model is reachable.

        Raises:
            EmbeddingModelNotReadyException: If Ollama or the model is unavailable.
        """
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
            models = [m["name"].split(":")[0] for m in payload.get("models", [])]
            model_name = self._model.split(":")[0]
            if model_name not in models:
                raise EmbeddingModelNotReadyException(
                    f"Embedding model '{self._model}' is not pulled in Ollama. "
                    f"Run: ollama pull {self._model}",
                    context={"model": self._model, "available": models},
                )
            logger.debug("Embedding health check passed", extra={"model": self._model})
            return True
        except EmbeddingModelNotReadyException:
            raise
        except httpx.ConnectError as exc:
            raise EmbeddingModelNotReadyException(
                f"Cannot reach Ollama at {self._base_url}: {exc}",
                context={"base_url": self._base_url},
            ) from exc
        except Exception as exc:
            raise EmbeddingModelNotReadyException(
                f"Embedding health check failed: {exc}"
            ) from exc

    async def model_info(self) -> dict[str, Any]:
        """Return basic model metadata."""
        return {
            "model": self._model,
            "provider": "ollama",
            "base_url": self._base_url,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _call_embeddings_api(self, text: str) -> list[float]:
        """
        Call POST /api/embeddings and return the embedding vector.

        Raises:
            EmbeddingException: On API errors.
        """
        start = time.monotonic()
        try:
            response = await self._client.post(
                "/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
        except httpx.TimeoutException as exc:
            raise EmbeddingException(
                f"Embedding request timed out after {self._settings.embedding_timeout}s",
                context={"model": self._model},
            ) from exc
        except httpx.ConnectError as exc:
            raise EmbeddingException(
                f"Cannot reach Ollama at {self._base_url}",
                context={"model": self._model},
            ) from exc

        elapsed_ms = (time.monotonic() - start) * 1000

        if response.status_code == 404:
            raise EmbeddingException(
                f"Embedding model '{self._model}' not found in Ollama. "
                f"Run: ollama pull {self._model}",
                context={"model": self._model},
            )

        if response.status_code != 200:
            raise EmbeddingException(
                f"Ollama embeddings API returned HTTP {response.status_code}",
                context={"status_code": response.status_code, "model": self._model},
            )

        try:
            data = response.json()
        except Exception as exc:
            raise EmbeddingException(
                "Failed to parse Ollama embeddings response as JSON"
            ) from exc

        embedding: list[float] = data.get("embedding", [])
        if not embedding:
            raise EmbeddingException(
                "Ollama returned an empty embedding vector.",
                context={"model": self._model},
            )

        logger.debug(
            "Embedding generated",
            extra={
                "model": self._model,
                "dimensions": len(embedding),
                "latency_ms": round(elapsed_ms, 2),
            },
        )

        return embedding
