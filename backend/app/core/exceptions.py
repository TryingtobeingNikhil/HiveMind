"""
app/core/exceptions.py
──────────────────────
Domain exception hierarchy for Open Deep Research.

All application exceptions inherit from AppException.
Global handlers in main.py translate these into consistent API error responses.

Response envelope:
    {
        "error": {
            "message": "Human-readable description",
            "type":    "ExceptionClassName",
            "code":    "MACHINE_READABLE_CODE"
        }
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any


# ── Error response model ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ErrorDetail:
    """Structured error payload returned to API consumers."""

    message: str
    type: str
    code: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "message": self.message,
            "type": self.type,
            "code": self.code,
        }
        if self.context:
            payload["context"] = self.context
        return payload


# ── Base exception ────────────────────────────────────────────────────────────


class AppException(Exception):
    """
    Base class for all application exceptions.

    All subclasses must set a human-readable ``message``, a ``code``
    (SCREAMING_SNAKE_CASE), and an appropriate HTTP status code.
    """

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context or {}

    @property
    def error_detail(self) -> ErrorDetail:
        return ErrorDetail(
            message=self.message,
            type=type(self).__name__,
            code=self.code,
            context=self.context,
        )

    def to_response(self) -> dict[str, Any]:
        return {"error": self.error_detail.to_dict()}


# ── Service exceptions ────────────────────────────────────────────────────────


class ServiceException(AppException):
    """Raised when a service-layer operation fails."""

    http_status: int = HTTPStatus.SERVICE_UNAVAILABLE
    code: str = "SERVICE_ERROR"


class ServiceNotReadyException(ServiceException):
    """Raised when a required service is not yet ready to accept requests."""

    http_status: int = HTTPStatus.SERVICE_UNAVAILABLE
    code: str = "SERVICE_NOT_READY"


# ── Validation exceptions ─────────────────────────────────────────────────────


class ValidationException(AppException):
    """Raised when input validation fails at the application layer."""

    http_status: int = HTTPStatus.UNPROCESSABLE_ENTITY
    code: str = "VALIDATION_ERROR"


# ── Ollama exceptions ─────────────────────────────────────────────────────────


class OllamaException(ServiceException):
    """Base class for all Ollama-related failures."""

    http_status: int = HTTPStatus.BAD_GATEWAY
    code: str = "OLLAMA_ERROR"


class OllamaConnectionError(OllamaException):
    """Raised when the Ollama server is unreachable."""

    http_status: int = HTTPStatus.BAD_GATEWAY
    code: str = "OLLAMA_CONNECTION_ERROR"


class OllamaTimeoutError(OllamaException):
    """Raised when an Ollama request exceeds the configured timeout."""

    http_status: int = HTTPStatus.GATEWAY_TIMEOUT
    code: str = "OLLAMA_TIMEOUT"


class OllamaModelNotFoundError(OllamaException):
    """Raised when the requested model is not available in Ollama."""

    http_status: int = HTTPStatus.NOT_FOUND
    code: str = "OLLAMA_MODEL_NOT_FOUND"


class OllamaGenerationError(OllamaException):
    """Raised when Ollama returns an error during text generation."""

    http_status: int = HTTPStatus.BAD_GATEWAY
    code: str = "OLLAMA_GENERATION_ERROR"


# ── Agent exceptions ──────────────────────────────────────────────────────────


class AgentException(AppException):
    """Base class for agent framework failures."""

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "AGENT_ERROR"


class AgentExecutionError(AgentException):
    """Raised when an agent's execute() method fails."""

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "AGENT_EXECUTION_ERROR"


class AgentNotReadyError(AgentException):
    """Raised when an agent is invoked before receiving a task."""

    http_status: int = HTTPStatus.CONFLICT
    code: str = "AGENT_NOT_READY"


# ── Not Found ─────────────────────────────────────────────────────────────────


class NotFoundException(AppException):
    """Generic 404 — resource not found."""

    http_status: int = HTTPStatus.NOT_FOUND
    code: str = "NOT_FOUND"


# ── Phase 2: Document & ingestion exceptions ──────────────────────────────────


class DocumentException(AppException):
    """Base class for document-related failures."""

    http_status: int = HTTPStatus.UNPROCESSABLE_ENTITY
    code: str = "DOCUMENT_ERROR"


class DocumentParsingException(DocumentException):
    """Raised when a document cannot be parsed (corrupt file, unsupported format, etc.)."""

    http_status: int = HTTPStatus.BAD_REQUEST
    code: str = "DOCUMENT_PARSING_ERROR"


class DocumentNotFoundException(DocumentException):
    """Raised when a document is not found in the metadata store."""

    http_status: int = HTTPStatus.NOT_FOUND
    code: str = "DOCUMENT_NOT_FOUND"


class IngestionException(DocumentException):
    """Raised when the document ingestion pipeline fails."""

    http_status: int = HTTPStatus.UNPROCESSABLE_ENTITY
    code: str = "INGESTION_ERROR"


# ── Phase 2: Embedding exceptions ─────────────────────────────────────────────


class EmbeddingException(ServiceException):
    """Base class for embedding generation failures."""

    http_status: int = HTTPStatus.BAD_GATEWAY
    code: str = "EMBEDDING_ERROR"


class EmbeddingModelNotReadyException(EmbeddingException):
    """Raised when the embedding model or provider is not available."""

    http_status: int = HTTPStatus.SERVICE_UNAVAILABLE
    code: str = "EMBEDDING_MODEL_NOT_READY"


# ── Phase 2: Vector store exceptions ─────────────────────────────────────────


class VectorStoreException(ServiceException):
    """Base class for vector store failures."""

    http_status: int = HTTPStatus.SERVICE_UNAVAILABLE
    code: str = "VECTOR_STORE_ERROR"


class VectorStoreNotReadyException(VectorStoreException):
    """Raised when the vector store is not initialised or reachable."""

    http_status: int = HTTPStatus.SERVICE_UNAVAILABLE
    code: str = "VECTOR_STORE_NOT_READY"


# ── Phase 2: Retrieval exceptions ─────────────────────────────────────────────


class RetrievalException(AppException):
    """Raised when the retrieval pipeline fails."""

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "RETRIEVAL_ERROR"


# ── Phase 2: Reranking exceptions ─────────────────────────────────────────────


class RerankingException(AppException):
    """Raised when the reranking step fails."""

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "RERANKING_ERROR"


# ── Phase 2: Token budget exceptions ─────────────────────────────────────────


class TokenBudgetException(AppException):
    """Raised when a token budget calculation fails or budget is exceeded."""

    http_status: int = HTTPStatus.UNPROCESSABLE_ENTITY
    code: str = "TOKEN_BUDGET_ERROR"


# ── Phase 2: LLM provider exceptions ─────────────────────────────────────────


class ProviderException(ServiceException):
    """Base class for LLM provider failures."""

    http_status: int = HTTPStatus.BAD_GATEWAY
    code: str = "PROVIDER_ERROR"


class ProviderNotReadyException(ProviderException):
    """Raised when the LLM provider is not available."""

    http_status: int = HTTPStatus.SERVICE_UNAVAILABLE
    code: str = "PROVIDER_NOT_READY"


# ── Phase 2: Database exceptions ─────────────────────────────────────────────


class DatabaseException(ServiceException):
    """Raised when a database operation fails."""

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "DATABASE_ERROR"


# ── Phase 3: Research orchestration exceptions ────────────────────────────────


class PlannerException(Exception):
    """Raised when PlannerAgent fails to produce a valid ResearchPlan.

    Common causes: LLM returned non-JSON, JSON missing required fields.
    Sessions that encounter this exception are immediately marked failed.
    """


class ResearchException(Exception):
    """Raised when ResearchAgent encounters a non-retrieval failure.

    Retrieval failures are handled gracefully (degraded result).
    This exception is reserved for unexpected agent-level errors.
    """


class CriticException(Exception):
    """Raised when CriticAgent fails to produce a valid CriticResult.

    In practice, CriticAgent catches its own failures and returns a
    fallback CriticResult. This exception may be raised for unexpected
    errors outside the normal parse-failure path.
    """


class ReportWriterException(Exception):
    """Raised when ReportWriterAgent fails to produce a valid ResearchReport.

    Sessions that encounter this exception are immediately marked failed.
    """


class SessionNotFoundException(Exception):
    """Raised when a research session is not found in the database.

    Used by ResearchRepository.get() and surfaced as HTTP 404 by routes.
    """


# ── Groq exceptions ───────────────────────────────────────────────────────────


class GroqException(ServiceException):
    """Base class for all Groq-related failures."""

    http_status: int = HTTPStatus.BAD_GATEWAY
    code: str = "GROQ_ERROR"


class GroqConnectionError(GroqException):
    """Raised when the Groq API is unreachable."""

    http_status: int = HTTPStatus.BAD_GATEWAY
    code: str = "GROQ_CONNECTION_ERROR"


class GroqTimeoutError(GroqException):
    """Raised when a Groq request exceeds the configured timeout."""

    http_status: int = HTTPStatus.GATEWAY_TIMEOUT
    code: str = "GROQ_TIMEOUT"


class GroqAuthenticationError(GroqException):
    """Raised when Groq returns HTTP 401 — the API key is invalid or missing."""

    http_status: int = HTTPStatus.UNAUTHORIZED
    code: str = "GROQ_AUTHENTICATION_ERROR"


class GroqRateLimitError(GroqException):
    """Raised when Groq returns HTTP 429 — rate limit exceeded."""

    http_status: int = HTTPStatus.TOO_MANY_REQUESTS
    code: str = "GROQ_RATE_LIMIT_ERROR"


class GroqGenerationError(GroqException):
    """Raised when Groq returns an error during text generation."""

    http_status: int = HTTPStatus.BAD_GATEWAY
    code: str = "GROQ_GENERATION_ERROR"


# ── Phase 4: Evaluator exceptions ─────────────────────────────────────────────


class EvaluatorException(Exception):
    """Raised when EvaluatorAgent encounters an unexpected agent-level error.

    In practice, EvaluatorAgent catches its own failures and returns a
    fail-safe EvaluationResult(sufficient=True). This exception is defined
    for completeness but should never propagate out of the agent.
    """
