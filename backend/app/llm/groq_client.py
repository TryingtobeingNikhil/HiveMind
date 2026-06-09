"""
app/llm/groq_client.py
──────────────────────
Async Groq HTTP client for Open Deep Research.

Design principles:
- All LLM interactions route through this class — agents never call Groq directly.
- Uses httpx.AsyncClient for fully non-blocking I/O (no Groq SDK).
- Implements exponential-backoff retries via tenacity.
- Raises typed GroqException subclasses so callers can handle failures precisely.
- Mirrors OllamaClient structure exactly: same method names, same log format,
  same retry guard, same _do_generate() separation.

EXECUTION MODEL CONSTRAINT
──────────────────────────────
This client is designed for single-active-request operation:

  - Only ONE generate() call is expected to be active system-wide at any time.
  - Retry logic is strictly sequential (one attempt at a time, no parallel bursts).
  - Callers (agents, services) must NOT invoke generate() concurrently.

The Phase 3 orchestrator enforces this by dispatching one agent at a time.
No locking or semaphores are needed here because concurrency is excluded
at the orchestration layer.

Groq REST API reference:
  Base URL:  https://api.groq.com/openai/v1
  Generate:  POST /chat/completions  (OpenAI-compatible)
  Models:    GET  /models

Public interface:
    client = GroqClient(settings)
    await client.health_check()
    models = await client.list_models()
    response = await client.generate(prompt="Hello", model="llama-3.1-8b-instant")
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
    GroqAuthenticationError,
    GroqConnectionError,
    GroqGenerationError,
    GroqRateLimitError,
    GroqTimeoutError,
)

logger = logging.getLogger(__name__)

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqClient:
    """
    Async HTTP client for the Groq REST API (OpenAI-compatible).

    EXECUTION MODEL
    ───────────────
    This client is intended for single-active-request use only.
    The system-wide assumption is that only ONE generate() call is in-flight
    at any moment. The Phase 3 orchestrator enforces this guarantee at the
    dispatch layer; no additional locking is required here.

    Lifecycle:
        The client should be initialised during application startup and closed
        on shutdown. Use ``async with GroqClient(settings) as client:`` or
        call ``await client.aclose()`` explicitly.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._default_model = settings.groq_model
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=settings.groq_timeout,
            write=30.0,
            pool=5.0,
        )
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            base_url=_GROQ_BASE_URL,
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        logger.debug(
            "GroqClient initialised",
            extra={"base_url": _GROQ_BASE_URL, "default_model": self._default_model},
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Close the underlying HTTP client and release connections."""
        await self._client.aclose()
        logger.debug("GroqClient closed")

    async def __aenter__(self) -> "GroqClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # ── Public API ────────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """
        Verify that the Groq API is reachable and the API key is valid.

        Returns:
            True if Groq responds with HTTP 200.

        Raises:
            GroqAuthenticationError: If the API key is invalid (HTTP 401).
            GroqConnectionError:     If the API is unreachable.
            GroqTimeoutError:        If the request times out.
        """
        try:
            start = time.monotonic()
            response = await self._client.get("/models")
            elapsed_ms = (time.monotonic() - start) * 1000

            if response.status_code == 401:
                raise GroqAuthenticationError(
                    "Groq API key is invalid or missing. "
                    "Check GROQ_API_KEY in your environment.",
                    context={"status_code": 401},
                )

            response.raise_for_status()
            logger.debug(
                "Groq health check passed",
                extra={"latency_ms": round(elapsed_ms, 2)},
            )
            return True

        except GroqAuthenticationError:
            raise
        except httpx.TimeoutException as exc:
            raise GroqTimeoutError(
                f"Groq health check timed out after {self._settings.groq_timeout}s"
            ) from exc
        except httpx.ConnectError as exc:
            raise GroqConnectionError(
                f"Cannot reach Groq API at {_GROQ_BASE_URL}. "
                "Check your network connection.",
                context={"base_url": _GROQ_BASE_URL},
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise GroqConnectionError(
                f"Groq health check failed with HTTP {exc.response.status_code}",
                context={"status_code": exc.response.status_code},
            ) from exc

    async def list_models(self) -> list[dict[str, Any]]:
        """
        Return a list of models available on the Groq API.

        Returns:
            List of model descriptor dicts from response["data"].

        Raises:
            GroqAuthenticationError: If the API key is invalid.
            GroqConnectionError:     If the API is unreachable.
            GroqTimeoutError:        If the request times out.
        """
        try:
            response = await self._client.get("/models")

            if response.status_code == 401:
                raise GroqAuthenticationError(
                    "Groq API key is invalid or missing.",
                    context={"status_code": 401},
                )

            response.raise_for_status()
            payload = response.json()
            models: list[dict[str, Any]] = payload.get("data", [])
            logger.debug("Listed Groq models", extra={"count": len(models)})
            return models

        except GroqAuthenticationError:
            raise
        except httpx.TimeoutException as exc:
            raise GroqTimeoutError("Timed out listing Groq models") from exc
        except httpx.ConnectError as exc:
            raise GroqConnectionError(
                f"Cannot reach Groq API at {_GROQ_BASE_URL}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise GroqConnectionError(
                f"Groq returned HTTP {exc.response.status_code} for /models"
            ) from exc

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Submit a chat completion request to Groq and return the response text.

        Args:
            prompt:  The user prompt to send.
            model:   Model name; defaults to ``settings.groq_model``.
            system:  Optional system prompt (sent as a system message).
            options: Optional generation parameters — supports ``temperature``
                     and ``max_tokens`` keys.

        Returns:
            The generated text string.

        Raises:
            GroqAuthenticationError: If the API key is invalid (HTTP 401).
            GroqRateLimitError:      If the rate limit is exceeded (HTTP 429).
            GroqGenerationError:     If Groq returns a generation error.
            GroqTimeoutError:        If generation exceeds the configured timeout.
            GroqConnectionError:     If the API is unreachable.
        """
        resolved_model = model or self._default_model
        opts = options or {}

        # Build message list (system message is optional)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": opts.get("temperature", 0.7),
            "max_tokens": opts.get("max_tokens", 4096),
            "stream": False,
        }

        logger.info(
            "Sending generation request",
            extra={
                "model": resolved_model,
                "prompt_length": len(prompt),
                "has_system": system is not None,
            },
        )

        try:
            # Sequential retry loop — retries on transient network failures AND
            # rate limit errors (HTTP 429). GroqAuthenticationError is NOT retried.
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(
                    (GroqConnectionError, GroqTimeoutError, GroqRateLimitError)
                ),
                stop=stop_after_attempt(self._settings.groq_max_retries + 1),
                wait=wait_exponential(
                    multiplier=self._settings.groq_retry_wait,
                    min=self._settings.groq_retry_wait,
                    max=60,  # rate limit resets within ~60s on free tier
                ),
                reraise=True,
            ):
                with attempt:
                    return await self._do_generate(payload, resolved_model)

        except RetryError as exc:
            raise GroqConnectionError(
                f"Groq generation failed after {self._settings.groq_max_retries} retries"
            ) from exc

        # Should not be reached due to reraise=True, but satisfies type checker
        raise GroqGenerationError("Unexpected error in generate()")  # pragma: no cover

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _do_generate(
        self, payload: dict[str, Any], model: str
    ) -> str:
        """Execute a single (non-retried) chat completion POST."""
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise GroqTimeoutError(
                f"Generation timed out after {self._settings.groq_timeout}s",
                context={"model": model},
            ) from exc
        except httpx.ConnectError as exc:
            raise GroqConnectionError(
                f"Cannot reach Groq API at {_GROQ_BASE_URL}",
                context={"model": model},
            ) from exc

        if response.status_code == 401:
            raise GroqAuthenticationError(
                "Groq API key is invalid or missing. Check GROQ_API_KEY.",
                context={"status_code": 401, "model": model},
            )

        if response.status_code == 429:
            raise GroqRateLimitError(
                "Groq rate limit exceeded. Retry after a short delay.",
                context={"status_code": 429, "model": model},
            )

        if response.status_code == 404:
            raise GroqGenerationError(
                f"Model '{model}' is not available on Groq. "
                "Check GROQ_MODEL in your environment.",
                context={"model": model, "status_code": 404},
            )

        if response.status_code != 200:
            raise GroqGenerationError(
                f"Groq returned HTTP {response.status_code}",
                context={"status_code": response.status_code, "model": model},
            )

        try:
            data = response.json()
        except Exception as exc:
            raise GroqGenerationError(
                "Failed to parse Groq response as JSON"
            ) from exc

        try:
            response_text: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise GroqGenerationError(
                "Groq response missing expected 'choices[0].message.content' path",
                context={"model": model},
            ) from exc

        logger.info(
            "Generation completed",
            extra={
                "model": model,
                "response_length": len(response_text),
                "usage_prompt_tokens": data.get("usage", {}).get("prompt_tokens"),
                "usage_completion_tokens": data.get("usage", {}).get("completion_tokens"),
            },
        )
        return response_text
