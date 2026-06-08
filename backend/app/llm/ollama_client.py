"""
app/llm/ollama_client.py
────────────────────────
Async Ollama HTTP client for Open Deep Research.

Design principles:
- All LLM interactions route through this class — agents never call Ollama directly.
- Uses httpx.AsyncClient for fully non-blocking I/O.
- Implements exponential-backoff retries via tenacity.
- Raises typed OllamaException subclasses so callers can handle failures precisely.

EXECUTION MODEL CONSTRAINT
──────────────────────────────
This client is designed for single-active-request operation:

  - Only ONE generate() call is expected to be active system-wide at any time.
  - Retry logic is strictly sequential (one attempt at a time, no parallel bursts).
  - Callers (agents, services) must NOT invoke generate() concurrently.

The Phase 3 orchestrator enforces this by dispatching one agent at a time.
No locking or semaphores are needed here because concurrency is excluded
at the orchestration layer.

Public interface:
    client = OllamaClient(settings)
    await client.health_check()
    models = await client.list_models()
    response = await client.generate(prompt="Hello", model="llama3.2:3b")
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings
from app.core.exceptions import (
    OllamaConnectionError,
    OllamaGenerationError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Async HTTP client for the Ollama REST API.

    EXECUTION MODEL
    ───────────────
    This client is intended for single-active-request use only.
    The system-wide assumption is that only ONE generate() call is in-flight
    at any moment. The Phase 3 orchestrator enforces this guarantee at the
    dispatch layer; no additional locking is required here.

    Lifecycle:
        The client should be initialised during application startup and closed
        on shutdown. Use ``async with OllamaClient(settings) as client:`` or
        call ``await client.aclose()`` explicitly.
    """

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._base_url = settings.ollama_api_url
        self._default_model = settings.ollama_model
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=settings.ollama_timeout,
            write=30.0,
            pool=5.0,
        )
        self._client: httpx.AsyncClient = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Content-Type": "application/json"},
        )
        logger.debug(
            "OllamaClient initialised",
            extra={"base_url": self._base_url, "default_model": self._default_model},
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Close the underlying HTTP client and release connections."""
        await self._client.aclose()
        logger.debug("OllamaClient closed")

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # ── Public API ────────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """
        Verify that the Ollama server is reachable.

        Returns:
            True if Ollama responds with HTTP 200.

        Raises:
            OllamaConnectionError: If the server is unreachable.
            OllamaTimeoutError:    If the request times out.
        """
        try:
            start = time.monotonic()
            response = await self._client.get("/api/tags")
            elapsed_ms = (time.monotonic() - start) * 1000
            response.raise_for_status()
            logger.debug(
                "Ollama health check passed",
                extra={"latency_ms": round(elapsed_ms, 2)},
            )
            return True
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama health check timed out after {self._settings.ollama_timeout}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self._base_url}. "
                "Ensure Ollama is running and OLLAMA_BASE_URL is correct.",
                context={"base_url": self._base_url},
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaConnectionError(
                f"Ollama health check failed with HTTP {exc.response.status_code}",
                context={"status_code": exc.response.status_code},
            ) from exc

    async def list_models(self) -> list[dict[str, Any]]:
        """
        Return a list of models available in the local Ollama installation.

        Returns:
            List of model descriptor dicts (name, modified_at, size, …).

        Raises:
            OllamaConnectionError: If the server is unreachable.
        """
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
            models: list[dict[str, Any]] = payload.get("models", [])
            logger.debug("Listed Ollama models", extra={"count": len(models)})
            return models
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Timed out listing Ollama models") from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self._base_url}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaConnectionError(
                f"Ollama returned HTTP {exc.response.status_code} for /api/tags"
            ) from exc

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> str:
        """
        Submit a generation request to Ollama and return the response text.

        Args:
            prompt:  The user prompt to send.
            model:   Model name; defaults to ``settings.ollama_model``.
            system:  Optional system prompt.
            options: Optional Ollama model parameters (temperature, top_p, …).
            stream:  If True, stream tokens (not yet surfaced to callers in Phase 1).

        Returns:
            The generated text string.

        Raises:
            OllamaModelNotFoundError: If the model is not pulled locally.
            OllamaGenerationError:    If Ollama returns an error payload.
            OllamaTimeoutError:       If generation exceeds the configured timeout.
            OllamaConnectionError:    If the server is unreachable.
        """
        resolved_model = model or self._default_model
        payload: dict[str, Any] = {
            "model": resolved_model,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        logger.info(
            "Sending generation request",
            extra={
                "model": resolved_model,
                "prompt_length": len(prompt),
                "stream": stream,
            },
        )

        try:
            # Sequential retry loop — each attempt waits for the previous one
            # to complete before starting. No parallel requests are issued.
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(
                    (OllamaConnectionError, OllamaTimeoutError)
                ),
                stop=stop_after_attempt(self._settings.ollama_max_retries + 1),
                wait=wait_exponential(
                    multiplier=self._settings.ollama_retry_wait,
                    min=self._settings.ollama_retry_wait,
                    max=30,
                ),
                reraise=True,
            ):
                with attempt:
                    return await self._do_generate(payload, resolved_model)

        except RetryError as exc:
            raise OllamaConnectionError(
                f"Ollama generation failed after {self._settings.ollama_max_retries} retries"
            ) from exc

        # Should not be reached due to reraise=True, but satisfies type checker
        raise OllamaGenerationError("Unexpected error in generate()")  # pragma: no cover

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _do_generate(
        self, payload: dict[str, Any], model: str
    ) -> str:
        """Execute a single (non-retried) generate POST."""
        try:
            response = await self._client.post("/api/generate", json=payload)
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Generation timed out after {self._settings.ollama_timeout}s",
                context={"model": model},
            ) from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self._base_url}",
                context={"model": model},
            ) from exc

        if response.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Model '{model}' is not available in Ollama. "
                f"Run: ollama pull {model}",
                context={"model": model},
            )

        if response.status_code != 200:
            raise OllamaGenerationError(
                f"Ollama returned HTTP {response.status_code}",
                context={"status_code": response.status_code, "model": model},
            )

        try:
            data = response.json()
        except Exception as exc:
            raise OllamaGenerationError(
                "Failed to parse Ollama response as JSON"
            ) from exc

        if "error" in data:
            raise OllamaGenerationError(
                f"Ollama generation error: {data['error']}",
                context={"model": model},
            )

        response_text: str = data.get("response", "")
        logger.info(
            "Generation completed",
            extra={
                "model": model,
                "response_length": len(response_text),
                "eval_count": data.get("eval_count"),
                "eval_duration_ms": round(data.get("eval_duration", 0) / 1_000_000, 2),
            },
        )
        return response_text
