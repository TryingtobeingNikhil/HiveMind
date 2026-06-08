"""
app/dependencies/providers.py
──────────────────────────────
FastAPI dependency providers for Open Deep Research.

All application-wide singletons are accessed through these providers, which
are passed to route handlers via ``Depends()``. This keeps routes decoupled
from global state and makes testing straightforward — override via
``app.dependency_overrides[get_memory_service] = lambda: mock_service``

Phase 1 providers (unchanged):
    get_settings()       → Settings
    get_ollama_client()  → OllamaClient   (from request.app.state)
    get_health_service() → HealthService
    get_logger()         → logging.Logger

Phase 2 providers (new):
    get_embedding_provider()  → BaseEmbeddingProvider
    get_vector_store()        → BaseVectorStore
    get_doc_repository()      → DocumentRepository
    get_metrics_repository()  → MetricsRepository
    get_reranker()            → BaseReranker | None
    get_memory_service()      → MemoryService
    get_token_budget_manager()→ TokenBudgetManager
    get_context_builder()     → ContextBuilder
    get_llm_manager()         → LLMExecutionManager
    get_llm_provider()        → BaseLLMProvider

Phase 3 providers (new):
    get_research_repository() → ResearchRepository  (singleton retriever)
    get_planner_agent()       → PlannerAgent         (singleton retriever)
    get_research_agent()      → ResearchAgent        (singleton retriever)
    get_critic_agent()        → CriticAgent          (singleton retriever)
    get_report_writer_agent() → ReportWriterAgent    (singleton retriever)
    get_research_orchestrator()→ ResearchOrchestrator (singleton retriever)

    All Phase 3 providers are RETRIEVERS ONLY — they read from app.state.
    The singletons are initialised in main.py lifespan() and shared across
    all requests. Providers never construct agents.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.llm.ollama_client import OllamaClient
from app.services.health_service import HealthService


# ── Settings ──────────────────────────────────────────────────────────────────

def settings_provider() -> Settings:
    """
    Provide the application Settings singleton.

    Wraps :func:`get_settings` so routes can use ``Depends(settings_provider)``
    rather than importing the cached function directly, enabling test overrides.
    """
    return get_settings()


SettingsDep = Annotated[Settings, Depends(settings_provider)]


# ── Ollama client ─────────────────────────────────────────────────────────────

def get_ollama_client(request: Request) -> OllamaClient:
    """
    Retrieve the OllamaClient singleton stored in application state.

    The client is initialised during the lifespan startup event in main.py
    and stored on ``app.state.ollama_client``.

    Raises:
        RuntimeError: If called before application startup completes.
    """
    client: OllamaClient | None = getattr(request.app.state, "ollama_client", None)
    if client is None:
        raise RuntimeError(
            "OllamaClient is not initialised. "
            "Ensure lifespan startup has completed before handling requests."
        )
    return client


OllamaClientDep = Annotated[OllamaClient, Depends(get_ollama_client)]


# ── Health service ────────────────────────────────────────────────────────────

def get_health_service(
    settings: SettingsDep,
    ollama_client: OllamaClientDep,
) -> HealthService:
    """
    Construct a HealthService per request.

    HealthService is lightweight (no I/O at construction time) so it is safe
    to instantiate per-request rather than as a singleton.
    """
    return HealthService(
        ollama_client=ollama_client,
        app_version=settings.app_version,
    )


HealthServiceDep = Annotated[HealthService, Depends(get_health_service)]


# ── Logger ────────────────────────────────────────────────────────────────────

def get_route_logger(request: Request) -> logging.Logger:
    """
    Return a logger scoped to the current request's route path.

    Usage in a route::

        @router.get("/example")
        async def example(log: LoggerDep) -> dict:
            log.info("Processing request")
    """
    route_path = getattr(request, "url", None)
    name = f"app.api.route.{route_path}" if route_path else "app.api.route"
    return logging.getLogger(name)


LoggerDep = Annotated[logging.Logger, Depends(get_route_logger)]


# ── Phase 2: Embedding provider ───────────────────────────────────────────────

def get_embedding_provider(request: Request):
    """Retrieve the OllamaEmbeddingProvider singleton from app.state."""
    from app.embedding.base import BaseEmbeddingProvider

    provider: BaseEmbeddingProvider | None = getattr(
        request.app.state, "embedding_provider", None
    )
    if provider is None:
        raise RuntimeError(
            "EmbeddingProvider is not initialised. "
            "Ensure lifespan startup has completed."
        )
    return provider


EmbeddingProviderDep = Annotated[object, Depends(get_embedding_provider)]


# ── Phase 2: Vector store ─────────────────────────────────────────────────────

def get_vector_store(request: Request):
    """Retrieve the ChromaVectorStore singleton from app.state."""
    from app.vectorstore.base import BaseVectorStore

    store: BaseVectorStore | None = getattr(request.app.state, "vector_store", None)
    if store is None:
        raise RuntimeError(
            "VectorStore is not initialised. "
            "Ensure lifespan startup has completed."
        )
    return store


VectorStoreDep = Annotated[object, Depends(get_vector_store)]


# ── Phase 2: Database repositories ───────────────────────────────────────────

def get_doc_repository(request: Request):
    """Retrieve the DocumentRepository from app.state."""
    from app.db.document_repository import DocumentRepository

    repo: DocumentRepository | None = getattr(request.app.state, "doc_repository", None)
    if repo is None:
        raise RuntimeError("DocumentRepository is not initialised.")
    return repo


DocRepositoryDep = Annotated[object, Depends(get_doc_repository)]


def get_metrics_repository(request: Request):
    """Retrieve the MetricsRepository from app.state."""
    from app.db.metrics_repository import MetricsRepository

    repo: MetricsRepository | None = getattr(
        request.app.state, "metrics_repository", None
    )
    if repo is None:
        raise RuntimeError("MetricsRepository is not initialised.")
    return repo


MetricsRepositoryDep = Annotated[object, Depends(get_metrics_repository)]


# ── Phase 2: Reranker ─────────────────────────────────────────────────────────

def get_reranker(request: Request):
    """Retrieve the optional reranker from app.state (may be None)."""
    return getattr(request.app.state, "reranker", None)


RerankerDep = Annotated[object, Depends(get_reranker)]


# ── Phase 2: Memory service ───────────────────────────────────────────────────

def get_memory_service(request: Request):
    """Retrieve the MemoryService singleton from app.state."""
    from app.memory.memory_service import MemoryService

    service: MemoryService | None = getattr(request.app.state, "memory_service", None)
    if service is None:
        raise RuntimeError(
            "MemoryService is not initialised. "
            "Ensure lifespan startup has completed."
        )
    return service


MemoryServiceDep = Annotated[object, Depends(get_memory_service)]


# ── Phase 2: Token budget manager ────────────────────────────────────────────

def get_token_budget_manager(request: Request):
    """Retrieve the TokenBudgetManager singleton from app.state."""
    from app.memory.token_budget import TokenBudgetManager

    manager: TokenBudgetManager | None = getattr(
        request.app.state, "token_budget_manager", None
    )
    if manager is None:
        raise RuntimeError("TokenBudgetManager is not initialised.")
    return manager


TokenBudgetManagerDep = Annotated[object, Depends(get_token_budget_manager)]


# ── Phase 2: Context builder ──────────────────────────────────────────────────

def get_context_builder(request: Request):
    """Retrieve the ContextBuilder singleton from app.state."""
    from app.memory.context_builder import ContextBuilder

    builder: ContextBuilder | None = getattr(
        request.app.state, "context_builder", None
    )
    if builder is None:
        raise RuntimeError("ContextBuilder is not initialised.")
    return builder


ContextBuilderDep = Annotated[object, Depends(get_context_builder)]


# ── Phase 2: LLM execution manager ───────────────────────────────────────────

def get_llm_manager(request: Request):
    """Retrieve the LLMExecutionManager singleton from app.state."""
    from app.llm.execution_manager import LLMExecutionManager

    manager: LLMExecutionManager | None = getattr(
        request.app.state, "llm_manager", None
    )
    if manager is None:
        raise RuntimeError("LLMExecutionManager is not initialised.")
    return manager


LLMManagerDep = Annotated[object, Depends(get_llm_manager)]


# ── Phase 2: LLM provider ─────────────────────────────────────────────────────

def get_llm_provider(request: Request):
    """Retrieve the BaseLLMProvider singleton from app.state."""
    from app.llm.base_provider import BaseLLMProvider

    provider: BaseLLMProvider | None = getattr(
        request.app.state, "llm_provider", None
    )
    if provider is None:
        raise RuntimeError("LLMProvider is not initialised.")
    return provider


LLMProviderDep = Annotated[object, Depends(get_llm_provider)]


# ── Phase 3: Research repository ──────────────────────────────────────────

def get_research_repository(request: Request):
    """
    Retrieve the ResearchRepository singleton from app.state.

    Initialised once in lifespan() and shared across all requests.
    """
    from app.db.research_repository import ResearchRepository

    repo: ResearchRepository | None = getattr(
        request.app.state, "research_repository", None
    )
    if repo is None:
        raise RuntimeError(
            "ResearchRepository is not initialised. "
            "Ensure lifespan startup has completed."
        )
    return repo


ResearchRepositoryDep = Annotated[object, Depends(get_research_repository)]


# ── Phase 3: Agent singletons ───────────────────────────────────────────────
# All agents are constructed ONCE in lifespan() with the shared LLM provider
# and execution manager. These providers are RETRIEVERS ONLY — they never
# construct agents.


def get_planner_agent(request: Request):
    """
    Retrieve the PlannerAgent singleton from app.state.

    The agent is constructed once at startup with the shared BaseLLMProvider
    and LLMExecutionManager, ensuring it always calls through the same lock.
    """
    from app.agents.planner_agent import PlannerAgent

    agent: PlannerAgent | None = getattr(request.app.state, "planner_agent", None)
    if agent is None:
        raise RuntimeError(
            "PlannerAgent is not initialised. "
            "Ensure lifespan startup has completed."
        )
    return agent


PlannerAgentDep = Annotated[object, Depends(get_planner_agent)]


def get_research_agent(request: Request):
    """
    Retrieve the ResearchAgent singleton from app.state.
    """
    from app.agents.research_agent import ResearchAgent

    agent: ResearchAgent | None = getattr(request.app.state, "research_agent", None)
    if agent is None:
        raise RuntimeError(
            "ResearchAgent is not initialised. "
            "Ensure lifespan startup has completed."
        )
    return agent


ResearchAgentDep = Annotated[object, Depends(get_research_agent)]


def get_critic_agent(request: Request):
    """
    Retrieve the CriticAgent singleton from app.state.
    """
    from app.agents.critic_agent import CriticAgent

    agent: CriticAgent | None = getattr(request.app.state, "critic_agent", None)
    if agent is None:
        raise RuntimeError(
            "CriticAgent is not initialised. "
            "Ensure lifespan startup has completed."
        )
    return agent


CriticAgentDep = Annotated[object, Depends(get_critic_agent)]


def get_report_writer_agent(request: Request):
    """
    Retrieve the ReportWriterAgent singleton from app.state.
    """
    from app.agents.report_writer_agent import ReportWriterAgent

    agent: ReportWriterAgent | None = getattr(
        request.app.state, "report_writer_agent", None
    )
    if agent is None:
        raise RuntimeError(
            "ReportWriterAgent is not initialised. "
            "Ensure lifespan startup has completed."
        )
    return agent


ReportWriterAgentDep = Annotated[object, Depends(get_report_writer_agent)]


# ── Phase 3: Orchestrator ────────────────────────────────────────────────────

def get_research_orchestrator(request: Request):
    """
    Retrieve the ResearchOrchestrator singleton from app.state.

    The orchestrator is constructed once at startup and wired with all
    agent singletons, the shared MemoryService, ResearchRepository, and
    LLMExecutionManager. It is safe to share across requests because Phase 3
    is synchronous — only one request runs at a time through the LLM lock.
    """
    from app.orchestration.orchestrator import ResearchOrchestrator

    orchestrator: ResearchOrchestrator | None = getattr(
        request.app.state, "research_orchestrator", None
    )
    if orchestrator is None:
        raise RuntimeError(
            "ResearchOrchestrator is not initialised. "
            "Ensure lifespan startup has completed."
        )
    return orchestrator


OrchestratorDep = Annotated[object, Depends(get_research_orchestrator)]


# ── Phase 4: EvaluatorAgent ──────────────────────────────────────────────────


def get_evaluator_agent(request: Request):
    """
    Retrieve the EvaluatorAgent singleton from app.state.

    The agent is constructed once at startup with the shared BaseLLMProvider
    and LLMExecutionManager. It is a RETRIEVER ONLY — never constructs agents.
    """
    from app.agents.evaluator_agent import EvaluatorAgent

    agent: EvaluatorAgent | None = getattr(
        request.app.state, "evaluator_agent", None
    )
    if agent is None:
        raise RuntimeError(
            "EvaluatorAgent is not initialised. "
            "Ensure lifespan startup has completed."
        )
    return agent


EvaluatorAgentDep = Annotated[object, Depends(get_evaluator_agent)]
