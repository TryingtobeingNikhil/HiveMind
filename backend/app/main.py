"""
app/main.py
────────────
Open Deep Research — FastAPI application factory.

Responsibilities:
  - Create and configure the FastAPI application instance.
  - Register the asynccontextmanager lifespan (startup + shutdown).
  - Mount the root API router.
  - Register global exception handlers.
  - Configure CORS middleware.
  - Expose top-level /health and /ready routes (framework-level shortcuts).

Phase 2 startup additions:
  - Initialise aiosqlite database + schema
  - Initialise DocumentRepository and MetricsRepository
  - Initialise OllamaEmbeddingProvider
  - Initialise ChromaVectorStore
  - Initialise CosineReranker
  - Initialise TokenBudgetManager and ContextBuilder
  - Initialise MemoryService
  - Initialise LLMExecutionManager
  - Initialise OllamaProvider

Phase 3 startup additions:
  - Initialise ResearchRepository (same SQLite connection)
  - Initialise PlannerAgent, ResearchAgent, CriticAgent, ReportWriterAgent
    as singletons sharing the Phase 2 LLM provider and execution manager
  - Initialise ResearchOrchestrator wiring all agents and services

Phase 4 startup additions:
  - Initialise EvaluatorAgent singleton
  - Pass EvaluatorAgent + Settings to ResearchOrchestrator for loop control
"""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.exceptions import AppException
from app.core.logging import configure_logging, get_logger
from app.core.telemetry import configure_telemetry
from app.llm.ollama_client import OllamaClient
from app.schemas.health import HealthResponse, HealthStatus

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Startup (Phase 1):
      1. Configure structured logging.
      2. Initialise the OllamaClient and store it on ``app.state``.
      3. Perform an initial Ollama connectivity probe (non-fatal on failure).

    Startup (Phase 2):
      4. Initialise SQLite database (create tables if needed).
      5. Initialise document and metrics repositories.
      6. Initialise embedding provider.
      7. Initialise ChromaDB vector store.
      8. Initialise reranker.
      9. Initialise MemoryService.
     10. Initialise TokenBudgetManager and ContextBuilder.
     11. Initialise LLMExecutionManager.
     12. Initialise OllamaProvider.

    Shutdown:
      - Close OllamaClient.
      - Close embedding provider HTTP client.
      - Close database connection.
    """
    settings: Settings = get_settings()

    # ── Startup ───────────────────────────────────────────────────────────────
    configure_logging(settings)
    configure_telemetry(settings)
    startup_log = get_logger(__name__)

    # Register graceful shutdown signals
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    def _signal_handler() -> None:
        startup_log.info("Received termination signal")
        shutdown_event.set()
        
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows fallback if needed

    startup_log.info(
        "Starting Open Deep Research",
        extra={
            "app_name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.app_env,
            "host": settings.host,
            "port": settings.port,
        },
    )

    # Phase 1: Initialise Ollama client
    ollama_client = OllamaClient(settings)
    app.state.ollama_client = ollama_client
    app.state.ready = False

    # Non-fatal connectivity probe at startup
    try:
        await ollama_client.health_check()
        startup_log.info(
            "Ollama connectivity verified",
            extra={"ollama_url": settings.ollama_api_url, "model": settings.ollama_model},
        )
    except Exception as exc:
        startup_log.warning(
            "Ollama not reachable at startup — readiness endpoint will reflect this",
            extra={"error": str(exc), "ollama_url": settings.ollama_api_url},
        )

    # Phase 2: Database
    from app.db.database import close_database, init_database
    from app.db.document_repository import DocumentRepository
    from app.db.metrics_repository import MetricsRepository

    db_conn = await init_database(settings.database_file_path)
    app.state.db_conn = db_conn

    doc_repo = DocumentRepository(db_conn)
    metrics_repo = MetricsRepository(db_conn)
    app.state.doc_repository = doc_repo
    app.state.metrics_repository = metrics_repo
    startup_log.info("Database repositories initialised")

    # Phase 2: Embedding provider
    from app.embedding.ollama_embedding import OllamaEmbeddingProvider

    embedding_provider = OllamaEmbeddingProvider(settings)
    app.state.embedding_provider = embedding_provider
    startup_log.info(
        "Embedding provider initialised",
        extra={"model": settings.embedding_model},
    )

    # Phase 2: Vector store
    from app.vectorstore.chroma_store import ChromaVectorStore

    vector_store = ChromaVectorStore(settings)
    await vector_store.create_collection()
    app.state.vector_store = vector_store
    startup_log.info(
        "ChromaDB vector store initialised",
        extra={
            "collection": settings.chroma_collection_name,
            "persist_dir": str(settings.chroma_persist_path),
        },
    )

    # Phase 2: Reranker
    from app.reranking.cosine_reranker import CosineReranker

    reranker = CosineReranker()
    app.state.reranker = reranker
    startup_log.info("CosineReranker initialised")

    # Phase 2: Memory service
    from app.memory.memory_service import MemoryService

    memory_service = MemoryService(
        settings=settings,
        embedding=embedding_provider,
        vector_store=vector_store,
        doc_repo=doc_repo,
        metrics_repo=metrics_repo,
        reranker=reranker,
    )
    app.state.memory_service = memory_service
    startup_log.info("MemoryService initialised")

    # Phase 2: Token budget manager
    from app.memory.token_budget import TokenBudgetManager

    token_budget_manager = TokenBudgetManager(settings)
    app.state.token_budget_manager = token_budget_manager

    # Phase 2: Context builder
    from app.memory.context_builder import ContextBuilder

    context_builder = ContextBuilder(settings, token_budget_manager)
    app.state.context_builder = context_builder
    startup_log.info("TokenBudgetManager and ContextBuilder initialised")

    # Phase 2: LLM execution manager
    from app.llm.execution_manager import LLMExecutionManager

    llm_manager = LLMExecutionManager(
        delay_seconds=settings.llm_call_delay_seconds
    )
    app.state.llm_manager = llm_manager
    startup_log.info("LLMExecutionManager initialised")

    # Phase 2 / Phase 3: LLM provider (Ollama or Groq, selected by LLM_PROVIDER)
    from app.llm.base_provider import BaseLLMProvider

    if settings.llm_provider == "groq":
        from app.llm.groq_client import GroqClient
        from app.llm.groq_provider import GroqProvider

        groq_client = GroqClient(settings)
        llm_provider: BaseLLMProvider = GroqProvider(groq_client)
        startup_log.info(
            "LLM provider initialised",
            extra={"provider": "groq", "model": settings.groq_model},
        )
    else:
        from app.llm.ollama_provider import OllamaProvider

        # Reuse the Phase 1 ollama_client already constructed above —
        # OllamaProvider wraps it without creating a new connection pool.
        llm_provider: BaseLLMProvider = OllamaProvider(ollama_client)
        startup_log.info(
            "LLM provider initialised",
            extra={"provider": "ollama", "model": settings.ollama_model},
        )

    app.state.llm_provider = llm_provider

    # Phase 3: Research repository (same SQLite connection as Phase 2)
    from app.db.research_repository import ResearchRepository

    research_repo = ResearchRepository(db_conn)
    app.state.research_repository = research_repo
    startup_log.info("ResearchRepository initialised")

    # Phase 3: Agent singletons
    # Each agent receives the shared llm_provider and llm_manager so that
    # all LLM calls across all agents go through the same execution lock.
    from app.agents.critic_agent import CriticAgent
    from app.agents.planner_agent import PlannerAgent
    from app.agents.report_writer_agent import ReportWriterAgent
    from app.agents.research_agent import ResearchAgent

    planner_agent = PlannerAgent(llm_provider, llm_manager)
    research_agent = ResearchAgent(llm_provider, llm_manager)
    critic_agent = CriticAgent(llm_provider, llm_manager)
    report_writer_agent = ReportWriterAgent(llm_provider, llm_manager)

    app.state.planner_agent = planner_agent
    app.state.research_agent = research_agent
    app.state.critic_agent = critic_agent
    app.state.report_writer_agent = report_writer_agent
    startup_log.info("Phase 3 agent singletons initialised (planner, researcher, critic, writer)")

    # Phase 4: EvaluatorAgent
    from app.agents.evaluator_agent import EvaluatorAgent

    evaluator_agent = EvaluatorAgent(
        provider=llm_provider,
        execution_manager=llm_manager,
    )
    app.state.evaluator_agent = evaluator_agent
    startup_log.info(
        "Phase 4 EvaluatorAgent initialised",
        extra={
            "max_research_iterations": settings.max_research_iterations,
            "confidence_threshold": settings.confidence_threshold,
            "evaluator_enabled": settings.evaluator_enabled,
        },
    )

    # Phase 3 + Phase 4: Orchestrator
    from app.orchestration.orchestrator import ResearchOrchestrator

    research_orchestrator = ResearchOrchestrator(
        planner=planner_agent,
        researcher=research_agent,
        critic=critic_agent,
        report_writer=report_writer_agent,
        memory_service=memory_service,
        research_repository=research_repo,
        execution_manager=llm_manager,
        skip_critic_for_low_complexity=settings.skip_critic_for_low_complexity,
        evaluator=evaluator_agent,
        settings=settings,
    )
    app.state.research_orchestrator = research_orchestrator
    startup_log.info("ResearchOrchestrator initialised (Phase 3 + Phase 4 autonomous loop)")

    # Phase 5: Task Queue
    from app.queue.redis_queue import ResearchTaskQueue

    task_queue = ResearchTaskQueue(settings)
    await task_queue.connect()
    app.state.task_queue = task_queue
    startup_log.info("ResearchTaskQueue initialised")

    app.state.ready = True
    startup_log.info("Application startup complete — Phase 4 autonomous research active")

    # ── Yield (application serves requests) ──────────────────────────────────
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    shutdown_log = get_logger(__name__)
    shutdown_log.info("Application shutdown initiated")
    app.state.ready = False

    # Close embedding provider
    try:
        await embedding_provider.aclose()
        shutdown_log.info("Embedding provider closed")
    except Exception as exc:
        shutdown_log.warning("Error closing embedding provider", extra={"error": str(exc)})

    # Close LLM provider client (Groq or Ollama)
    try:
        provider_client = getattr(app.state.llm_provider, "_client", None)
        if provider_client and hasattr(provider_client, "aclose"):
            await provider_client.aclose()
            shutdown_log.info("LLM provider client closed")
    except Exception as exc:
        shutdown_log.warning("Error closing LLM provider client", extra={"error": str(exc)})

    # Close Ollama client
    await ollama_client.aclose()
    shutdown_log.info("OllamaClient closed")

    # Close Redis task queue
    if getattr(app.state, "task_queue", None):
        await app.state.task_queue.disconnect()

    # Close database
    await close_database(db_conn)

    shutdown_log.info(
        "Application shutdown complete",
        extra={"app_name": settings.app_name},
    )


# ── Application factory ───────────────────────────────────────────────────────


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        settings: Optional Settings override (used in tests).

    Returns:
        A fully configured :class:`FastAPI` instance.
    """
    cfg = settings or get_settings()

    app = FastAPI(
        title="Open Deep Research API",
        description=(
            "A fully local, multi-agent AI research platform powered by Ollama. "
            "Phase 2: Memory & RAG Foundation."
        ),
        version=cfg.app_version,
        docs_url="/docs" if cfg.is_development else None,
        redoc_url="/redoc" if cfg.is_development else None,
        openapi_url="/openapi.json" if cfg.is_development else None,
        lifespan=lifespan,
    )

    _register_middleware(app, cfg)
    _register_exception_handlers(app)
    _register_routes(app)

    if cfg.otel_enabled:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)

    return app


def _register_middleware(app: FastAPI, settings: Settings) -> None:
    """Register all middleware in correct stack order (last added = outermost)."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers returning consistent error envelopes."""

    @app.exception_handler(AppException)
    async def handle_app_exception(
        request: Request, exc: AppException
    ) -> JSONResponse:
        """Handle all domain exceptions from the AppException hierarchy."""
        logger.error(
            "Application exception",
            extra={
                "code": exc.code,
                "type": type(exc).__name__,
                "error_message": exc.message,
                "path": str(request.url),
                "context": exc.context,
            },
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_response(),
        )

    @app.exception_handler(ValidationError)
    async def handle_pydantic_validation(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors with the standard error envelope."""
        logger.warning(
            "Validation error",
            extra={"path": str(request.url), "error_details": exc.errors()},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "message": "Request validation failed",
                    "type": "ValidationError",
                    "code": "VALIDATION_ERROR",
                    "context": {"errors": exc.errors()},
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler — prevents raw stack traces leaking to clients."""
        logger.exception(
            "Unhandled exception",
            extra={"path": str(request.url), "error": str(exc)},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": "An unexpected internal error occurred.",
                    "type": "InternalServerError",
                    "code": "INTERNAL_ERROR",
                    "context": {},
                }
            },
        )


def _register_routes(app: FastAPI) -> None:
    """Mount all routers onto the application."""

    # ── Top-level convenience endpoints ───────────────────────────────────────
    # These live at the root (not under /api/v1) for infrastructure consumers
    # (load balancers, Docker health checks) that don't know the API version.

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Infrastructure"],
        summary="Application liveness",
        include_in_schema=True,
    )
    async def root_health() -> HealthResponse:
        """Root liveness check — always returns 200 if the process is alive."""
        return HealthResponse(status=HealthStatus.HEALTHY)

    @app.get(
        "/ready",
        tags=["Infrastructure"],
        summary="Application readiness (root shortcut)",
        include_in_schema=True,
    )
    async def root_ready(request: Request) -> JSONResponse:
        """
        Root readiness shortcut — delegates to the v1 readiness service.

        Exists so Docker HEALTHCHECK and Kubernetes probes can target /ready
        without specifying the API version.
        """
        from app.dependencies.providers import get_health_service, get_ollama_client, settings_provider

        settings = settings_provider()
        ollama_client = get_ollama_client(request)
        health_service = get_health_service(settings=settings, ollama_client=ollama_client)
        task_queue = getattr(request.app.state, "task_queue", None)
        response = await health_service.check_readiness(task_queue=task_queue)
        http_status = 200 if response.is_ready else 503
        return JSONResponse(
            status_code=http_status,
            content=response.model_dump(mode="json"),
        )

    # ── Versioned API routes ──────────────────────────────────────────────────
    app.include_router(api_router)


# ── Application instance ──────────────────────────────────────────────────────
# This module-level instance is what uvicorn targets: ``uvicorn app.main:app``
app = create_app()
