# Open Deep Research

Open Deep Research is a fully local, production-grade autonomous AI research system that synthesises comprehensive reports from large document collections. Unlike a standard chatbot that answers queries in a single generative pass based on its training weights, this system decomposes complex questions into smaller tasks, retrieves factual evidence from a local vector database, critiques its own findings, and autonomously loops to fill information gaps until a defined confidence threshold is met.

---

## Architecture

1. User interface layer provides interaction channels including a CLI, a REST API, and an SSE stream for real-time state updates.
2. FastAPI gateway handles routing, request validation, and session management using Pydantic schemas.
3. Orchestration layer coordinates the sequential multi-agent pipeline and manages the autonomous evaluation loop.
4. Agent layer executes discrete tasks through specialized agents including the Planner, Researcher, Critic, Writer, and Evaluator.
5. Infrastructure layer manages persistence and observability using a SQLite memory service, Redis for task queuing, OpenTelemetry for tracing, and Docker for containerisation.

One LLM call at a time, enforced by LLMExecutionManager.

---

## Key engineering decisions

**Sequential execution over parallel agents**
The orchestrator executes research tasks and agent calls one at a time through a global lock. This was necessary to run multiple agents on local hardware without memory exhaustion or context thrashing. The tradeoff is slower overall session completion times compared to asynchronous, parallelised pipelines.

**ChromaDB over FAISS**
ChromaDB was selected as the vector store for document embeddings instead of FAISS. It provides out-of-the-box metadata filtering and persistent storage without requiring manual index management. The tradeoff is slightly lower raw similarity search performance at extreme scale compared to FAISS.

**Deterministic confidence scoring**
Confidence scores are calculated mathematically as the mean of document retrieval scores rather than relying on the LLM to self-report its certainty. This prevents the LLM from hallucinating high confidence on weak evidence. The tradeoff is that the score reflects retrieval quality, not necessarily the reasoning quality of the synthesised findings.

**Cloud LLM (Groq) over local**
The system defaults to using the Groq API (llama-3.1-8b-instant) instead of local Ollama execution when configured. This decision was made to accommodate deployment on constrained hardware like an M2 Air with 8GB RAM. The explicit tradeoff is introducing a network dependency and privacy concerns in exchange for viable inference speeds.

**Redis for task queue only**
Redis is deployed strictly to manage the background queuing of incoming research sessions. It provides persistence across application restarts and decouples request intake from execution. The tradeoff is adding an entire infrastructural dependency for a single use case, increasing operational complexity.

**SSE over WebSockets**
Server-Sent Events (SSE) were chosen for streaming session progress back to the client instead of WebSockets. SSE is perfectly suited for this one-directional event broadcast and requires significantly less connection overhead. The tradeoff is that the client cannot send interactive feedback or interrupt the agent mid-stream over the same connection.

---

## Tech stack

| Component | Technology | Why chosen |
|---|---|---|
| Web framework | FastAPI | Asynchronous routing and automatic OpenAPI validation via Pydantic. |
| LLM provider | Groq / Ollama | Abstracted provider layer allows swapping local models for fast cloud inference. |
| Embeddings | nomic-embed-text | Local embedding generation via Ollama for privacy-preserving document ingestion. |
| Vector DB | ChromaDB | Simple persistent storage and metadata filtering for retrieved chunks. |
| Metadata DB | aiosqlite | Lightweight, asynchronous relational storage for session states and metrics. |
| Task queue | Redis | Persistent queueing of research tasks to prevent overload during traffic spikes. |
| Tracing | OpenTelemetry | Standardised distributed observability for monitoring agent execution pipelines. |
| Containerisation | Docker Compose | Reproducible, isolated environments for the backend, Ollama, and Redis. |
| HTTP client | httpx | Asynchronous HTTP requests required for communicating with the Ollama API. |
| Retry logic | Tenacity | Resilient decorators for handling transient network failures in LLM interactions. |

---

## How it works

1. The user submits a research query to the API which establishes a new session identifier.
2. The Planner agent decomposes the original query into a structured list of actionable research tasks.
3. The research loop begins by passing these tasks sequentially to the Researcher agent, which retrieves relevant document excerpts and synthesises factual findings.
4. The Critic agent evaluates the findings for completeness, contradictions, and overall quality, providing structured feedback.
5. The Writer agent synthesises the original plan, the accumulated findings, and the critic's feedback into a comprehensive final report.
6. The Evaluator agent assesses the final report against a predefined confidence threshold computed from the retrieval scores.
7. If confidence is below the threshold and iterations are remaining, gap tasks are generated and the research loop repeats.
8. Once the report meets the quality threshold or the maximum iterations are exhausted, the final report is delivered to the user.

---

## Project structure

```text
app/
├── agents/        # Contains the core LLM-driven entities (Planner, Researcher, Critic, Writer, Evaluator)
├── api/           # Defines the REST API endpoints, routing, and SSE streaming logic
├── core/          # Houses application-wide configuration, exceptions, logging, and telemetry setup
├── db/            # Manages asynchronous SQLite connections, repositories, and schema migrations
├── embedding/     # Handles the generation of vector embeddings using local or external providers
├── llm/           # Provides the unified execution manager, rate limiting, and provider clients
├── memory/        # Orchestrates document retrieval, context building, and token budget management
├── orchestration/ # Coordinates the sequential multi-agent execution pipeline and evaluation loop
├── queue/         # Implements the persistent Redis-based task queue for session management
├── reranking/     # Contains logic for re-ordering retrieved chunks based on cosine similarity
├── schemas/       # Defines all Pydantic models used for data validation and agent communication
└── vectorstore/   # Manages the interaction with ChromaDB for storing and querying document embeddings
```

---

## Quick start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Groq API key (free tier sufficient)

### Local development
```bash
git clone https://github.com/TryingtobeingNikhil/HiveMind.git
cd backend
cp .env.example .env
# add GROQ_API_KEY to .env
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Docker compose (full stack)
```bash
docker compose up --build
```

### Run tests
```bash
cd backend && python -m pytest tests/ -v
```

---

## API reference

| Method | Path | Description | Response |
|---|---|---|---|
| GET | `/health` | Root liveness check | `{"status": "healthy"}` |
| GET | `/ready` | Root readiness shortcut | `{"status": "ready", ...}` |
| POST | `/api/v1/research/start` | Start a research session | `202 Accepted` with session metadata |
| GET | `/api/v1/research/history` | List past sessions | `200 OK` with paginated list of sessions |
| GET | `/api/v1/research/{session_id}` | Get full WorkflowState | `200 OK` with full session state |
| GET | `/api/v1/research/{session_id}/status` | Get lightweight status | `200 OK` with status and stage |
| GET | `/api/v1/research/{session_id}/report` | Get the final report | `200 OK` with report or `404/422` |
| GET | `/api/v1/research/{session_id}/stream` | Stream updates | SSE stream yielding WorkflowState JSON |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Which LLM provider to use (`ollama` or `groq`). |
| `GROQ_API_KEY` | `None` | Groq API key. Required when LLM_PROVIDER=groq. |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name. |
| `MAX_RESEARCH_ITERATIONS` | `2` | Maximum research loop iterations. |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum mean confidence score to consider research sufficient. |
| `EVALUATOR_ENABLED` | `True` | If False, skip evaluation loop entirely. |
| `LLM_CALL_DELAY_SECONDS` | `0.0` | Seconds to wait after each LLM call completes. |
| `SKIP_CRITIC_FOR_LOW_COMPLEXITY` | `True` | Skip CRITIQUING stage for low complexity plans. |
| `OTEL_ENABLED` | `False` | If True, enable OpenTelemetry tracing. |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL for the task queue. |

---

## What I would do differently

* Add authentication before any public deployment.
* Parallel agent execution when hardware allows.
* WebSockets for bidirectional research interaction.
* FAISS at document scale > 100k chunks.
