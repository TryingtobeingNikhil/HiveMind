"""
app/core/config.py
──────────────────
Centralised configuration using Pydantic Settings.
All configuration is sourced from environment variables (with .env file support).

Usage:
    from app.core.config import get_settings
    settings = get_settings()

SYSTEM-WIDE EXECUTION MODEL CONSTRAINT
───────────────────────────────────────
All services in Open Deep Research must operate under a
SINGLE ACTIVE EXECUTION PIPELINE assumption:

  - No service should require concurrent execution of LLM calls.
  - No service should require concurrent execution of agent tasks.
  - The Phase 3 orchestrator dispatches agents one at a time.
  - Only ONE LLM call (via OllamaClient.generate) is active at any moment.

This constraint is enforced architecturally at the orchestration layer.
Service and agent authors must NOT build components that assume or require
parallel LLM or agent execution.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings.

    Values are loaded (in priority order) from:
      1. Environment variables
      2. .env file at the backend root
      3. Defaults defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = Field(default="open-deep-research", alias="APP_NAME")
    app_env: Literal["development", "production", "testing"] = Field(
        default="development", alias="APP_ENV"
    )
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT", ge=1, le=65535)
    workers: int = Field(default=1, alias="WORKERS", ge=1)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_format: Literal["text", "json"] = Field(default="text", alias="LOG_FORMAT")

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="llama3.2:3b", alias="OLLAMA_MODEL")
    ollama_timeout: float = Field(default=120.0, alias="OLLAMA_TIMEOUT", gt=0)
    ollama_max_retries: int = Field(default=3, alias="OLLAMA_MAX_RETRIES", ge=0)
    ollama_retry_wait: float = Field(default=2.0, alias="OLLAMA_RETRY_WAIT", ge=0)

    # ── API ───────────────────────────────────────────────────────────────────
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        alias="CORS_ORIGINS",
    )
    cors_allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")

    # ── Phase 2: Tokenizer ────────────────────────────────────────────────────
    # HuggingFace model name for AutoTokenizer.
    # Must match (or be compatible with) the embedding model.
    # Default "bert-base-uncased" is compatible with nomic-embed-text.
    # For Qwen-based systems: "Qwen/Qwen2-7B"
    # For Llama-based systems: "meta-llama/Llama-3.2-3B"
    tokenizer_model: str = Field(
        default="bert-base-uncased", alias="TOKENIZER_MODEL"
    )

    # ── Phase 2: Chunking ─────────────────────────────────────────────────────
    chunk_token_size: int = Field(default=512, alias="CHUNK_TOKEN_SIZE", ge=64, le=4096)
    chunk_overlap_tokens: int = Field(
        default=50, alias="CHUNK_OVERLAP_TOKENS", ge=0, le=512
    )

    # ── Phase 2: Embedding ────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="nomic-embed-text", alias="EMBEDDING_MODEL"
    )
    embedding_timeout: float = Field(
        default=30.0, alias="EMBEDDING_TIMEOUT", gt=0
    )
    embedding_batch_size: int = Field(
        default=32, alias="EMBEDDING_BATCH_SIZE", ge=1, le=256
    )

    # ── Phase 2: Vector store ─────────────────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./data/chroma", alias="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field(
        default="documents", alias="CHROMA_COLLECTION_NAME"
    )

    # ── Phase 2: Retrieval ────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K", ge=1, le=100)
    retrieval_candidate_k: int = Field(
        default=20, alias="RETRIEVAL_CANDIDATE_K", ge=1, le=200
    )
    retrieval_score_threshold: float = Field(
        default=0.0, alias="RETRIEVAL_SCORE_THRESHOLD", ge=0.0, le=1.0
    )
    reranker_enabled: bool = Field(default=True, alias="RERANKER_ENABLED")

    # ── Phase 2: Token budget ─────────────────────────────────────────────────
    max_context_tokens: int = Field(
        default=4096, alias="MAX_CONTEXT_TOKENS", ge=512
    )
    reserved_output_tokens: int = Field(
        default=512, alias="RESERVED_OUTPUT_TOKENS", ge=64
    )

    # ── Phase 2: Storage ──────────────────────────────────────────────────────
    database_path: str = Field(
        default="./data/documents.db", alias="DATABASE_PATH"
    )
    uploads_dir: str = Field(
        default="./data/uploads", alias="UPLOADS_DIR"
    )

    # ── Phase 3: Orchestration ────────────────────────────────────────────────
    llm_call_delay_seconds: float = Field(
        default=0.0,
        alias="LLM_CALL_DELAY_SECONDS",
        ge=0.0,
        le=10.0,
        description=(
            "Seconds to wait after each LLM call completes before releasing "
            "the execution lock. Set to 0.0 (default) for no delay. "
            "Increase only if hitting provider rate limits in testing."
        ),
    )

    skip_critic_for_low_complexity: bool = Field(
        default=True,
        alias="SKIP_CRITIC_FOR_LOW_COMPLEXITY",
        description=(
            "If True, sessions where PlannerAgent estimates complexity='low' "
            "skip the CRITIQUING stage entirely and proceed directly to "
            "REPORTING with an auto-approved CriticResult. "
            "Saves one LLM call per low-complexity session."
        ),
    )

    # ── Phase 4: Autonomous Research ─────────────────────────────────────────
    max_research_iterations: int = Field(
        default=2,
        alias="MAX_RESEARCH_ITERATIONS",
        ge=1,
        le=5,
        description=(
            "Maximum research loop iterations. "
            "1 = single pass (Phase 3 behaviour). "
            "2 = one improvement pass (default). "
            "Hard ceiling regardless of confidence."
        ),
    )
    confidence_threshold: float = Field(
        default=0.5,
        alias="CONFIDENCE_THRESHOLD",
        ge=0.0,
        le=1.0,
        description=(
            "Minimum mean confidence score to consider research sufficient. "
            "Computed from ResearchResult.confidence values, not LLM self-report."
        ),
    )
    evaluator_enabled: bool = Field(
        default=True,
        alias="EVALUATOR_ENABLED",
        description=(
            "If False, skip evaluation loop entirely — "
            "behaves exactly like Phase 3. "
            "Useful for debugging and rate limit conservation."
        ),
    )

    # ── Phase 5: Production Infrastructure ────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL",
        description="Redis connection URL for the task queue.",
    )
    sse_poll_interval_seconds: float = Field(
        default=1.0,
        alias="SSE_POLL_INTERVAL_SECONDS",
        ge=0.1,
        le=5.0,
        description="How often the SSE endpoint polls the database for state changes.",
    )
    otel_enabled: bool = Field(
        default=False,
        alias="OTEL_ENABLED",
        description="If True, enable OpenTelemetry tracing.",
    )
    otel_endpoint: str = Field(
        default="http://localhost:4318/v1/traces",
        alias="OTEL_ENDPOINT",
        description="OTLP HTTP endpoint for traces (e.g. Jaeger).",
    )
    otel_service_name: str = Field(
        default="open-deep-research",
        alias="OTEL_SERVICE_NAME",
        description="Service name reported to the tracing backend.",
    )

    # ── Phase 3: Groq Provider ────────────────────────────────────────────────
    llm_provider: Literal["ollama", "groq"] = Field(
        default="ollama",
        alias="LLM_PROVIDER",
        description="Which LLM provider to use. 'ollama' = local. 'groq' = Groq cloud API.",
    )
    groq_api_key: str | None = Field(
        default=None,
        alias="GROQ_API_KEY",
        description="Groq API key. Required when LLM_PROVIDER=groq.",
    )
    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        alias="GROQ_MODEL",
        description="Groq model name. llama-3.1-8b-instant is fastest on free tier.",
    )
    groq_timeout: float = Field(
        default=60.0,
        alias="GROQ_TIMEOUT",
        gt=0,
        description="Seconds before a Groq request times out.",
    )
    groq_max_retries: int = Field(
        default=2,
        alias="GROQ_MAX_RETRIES",
        ge=0,
        description="Max retry attempts for transient Groq failures.",
    )
    groq_retry_wait: float = Field(
        default=5.0,
        alias="GROQ_RETRY_WAIT",
        ge=0,
        description="Base wait seconds between Groq retries.",
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Accept both JSON array strings and actual lists."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                # Treat as comma-separated string
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @model_validator(mode="after")
    def validate_env_consistency(self) -> "Settings":
        """Enforce environment-specific constraints."""
        if self.app_env == "production":
            if self.debug:
                raise ValueError("DEBUG must be False in production environment.")
            if self.log_format != "json":
                raise ValueError("LOG_FORMAT must be 'json' in production environment.")

        if self.chunk_overlap_tokens >= self.chunk_token_size:
            raise ValueError(
                "CHUNK_OVERLAP_TOKENS must be less than CHUNK_TOKEN_SIZE."
            )
        if self.llm_provider == "groq" and self.groq_api_key is None:
            raise ValueError(
                "GROQ_API_KEY must be set when LLM_PROVIDER=groq"
            )
        return self

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def is_testing(self) -> bool:
        return self.app_env == "testing"

    @property
    def ollama_api_url(self) -> str:
        """Normalised Ollama API base (no trailing slash)."""
        return self.ollama_base_url.rstrip("/")

    @property
    def chroma_persist_path(self) -> Path:
        """Resolved absolute path for ChromaDB persistence directory."""
        return Path(self.chroma_persist_dir).resolve()

    @property
    def database_file_path(self) -> Path:
        """Resolved absolute path for the SQLite database file."""
        return Path(self.database_path).resolve()

    @property
    def uploads_dir_path(self) -> Path:
        """Resolved absolute path for uploaded file storage."""
        return Path(self.uploads_dir).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings singleton.

    The cache is intentionally module-level so the same instance is returned
    on every call. In tests, use dependency_overrides to substitute a
    test-specific Settings instance.
    """
    return Settings()
