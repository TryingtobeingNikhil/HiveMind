"""
app/schemas/common.py
─────────────────────
Shared Pydantic response envelopes used across all API routes.

These types enforce a consistent API contract throughout the application,
making it easy for clients to parse both success and error responses.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """
    Generic success response envelope.

    Example:
        {
            "data": { ... },
            "meta": { "version": "v1" }
        }
    """

    data: T
    meta: dict[str, Any] = Field(default_factory=dict)


class ErrorPayload(BaseModel):
    """Inner error object embedded in :class:`ErrorResponse`."""

    message: str = Field(..., description="Human-readable error description")
    type: str = Field(..., description="Exception class name")
    code: str = Field(..., description="Machine-readable error code (SCREAMING_SNAKE_CASE)")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional additional context about the error",
    )


class ErrorResponse(BaseModel):
    """
    Standard error response envelope.

    Example:
        {
            "error": {
                "message": "Ollama server is unreachable",
                "type":    "OllamaConnectionError",
                "code":    "OLLAMA_CONNECTION_ERROR",
                "context": {}
            }
        }
    """

    error: ErrorPayload
