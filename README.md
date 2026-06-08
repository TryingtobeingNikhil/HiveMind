# Open Deep Research

A fully local, multi-agent AI research platform powered by Ollama.

## Architecture

```
Open Deep Research
└── backend/          ← Phase 1 (this phase)
    ├── app/
    │   ├── api/      ← FastAPI versioned routing
    │   ├── core/     ← Config, logging, exceptions
    │   ├── agents/   ← Agent framework (BaseAgent)
    │   ├── llm/      ← Ollama client
    │   ├── schemas/  ← Pydantic models
    │   ├── services/ ← Business logic
    │   └── dependencies/ ← DI providers
    └── tests/
```

## Phase Roadmap

| Phase | Focus |
|-------|-------|
| 1 | Foundation & Infrastructure |
| 2 | RAG + Vector DB |
| 3 | Orchestrator & Reflection Loops |
| 4 | Research Workflows |
| 5 | Frontend |

## Execution Model

> **System-wide constraint — all contributors must read this.**

Open Deep Research uses a **strictly sequential, single-pipeline execution model**:

- **One agent active at a time.** The Phase 3 orchestrator dispatches agents one at a time in a defined sequence. No two agents run concurrently.
- **One LLM call at a time.** Only a single `OllamaClient.generate()` call is in-flight across the entire system at any moment.
- **No distributed workers.** There is no multi-process or multi-thread agent execution model. The orchestrator runs in a single async event loop.
- **No concurrency primitives in agents.** Agents must not spawn background tasks, use locks/semaphores, or assume shared state with other simultaneously-active agents.

This constraint is enforced architecturally at the orchestration layer (Phase 3). Service and agent authors do **not** need to implement any additional synchronisation.

Expected log pattern for a multi-agent pipeline:

```
[ORCHESTRATOR] Starting task
[AGENT] Planner executed
[AGENT] Research executed
[AGENT] Summariser executed
[ORCHESTRATOR] Task complete
```

## Quickstart (Docker)

```bash
# Copy environment config
cp backend/.env.example backend/.env

# Start all services (backend + Ollama)
docker compose -f backend/docker-compose.yml up --build

# Verify
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/ready
```

## Quickstart (Local Development)

**Prerequisites**: Python 3.12+, [Ollama](https://ollama.ai) installed and running

```bash
cd backend

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Configure environment
cp .env.example .env
# Edit .env as needed

# Pull a model (if not already done)
ollama pull llama3.2:3b

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Running Tests

```bash
cd backend
pytest tests/ -v --tb=short
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Application health check |
| GET | `/ready` | Readiness check (Ollama connectivity) |
| GET | `/api/v1/health` | v1 health |
| GET | `/api/v1/ready` | v1 readiness |
| POST | `/api/v1/agents/echo` | Echo agent (framework validation) |

## Environment Variables

See [`backend/.env.example`](backend/.env.example) for all configurable variables.

## Technology Stack

- **Runtime**: Python 3.12+
- **Framework**: FastAPI + Uvicorn
- **LLM Backend**: Ollama
- **Config**: Pydantic Settings
- **Logging**: Python stdlib logging (JSON in production)
- **Testing**: pytest + pytest-asyncio + httpx
- **Containerization**: Docker + Docker Compose
