<div align="center">

<img src="https://img.shields.io/badge/HiveMind-Autonomous%20Deep%20Research-2D9CDB?style=for-the-badge&logoColor=white" alt="HiveMind">

# 🧠 HiveMind

### *Ask a hard question. Get a researched, critiqued, and verified report — autonomously.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-Agent%20Orchestration-1C3C3C?style=flat-square)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=flat-square)](https://trychroma.com)
[![Redis](https://img.shields.io/badge/Redis-Task%20Queue-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Tracing-425CC7?style=flat-square)](https://opentelemetry.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

**Not a chatbot. Not a single-pass RAG system. A self-correcting research engine that knows when it doesn't know enough.**

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [How It Works](#-how-it-works) · [Engineering Decisions](#-key-engineering-decisions) · [API Reference](#-api-reference)

</div>

---

## 🎯 The Problem

You paste a complex question into ChatGPT. It answers confidently — from training weights, without reading your documents, without checking if the answer is complete, without knowing what it missed.

Standard RAG is only marginally better. It retrieves some chunks, generates once, and returns. There's no quality check. No "did I actually answer this?" No loop.

**HiveMind fixes this.**

---

## ✨ What HiveMind Does

> Submit a research query against your document collection. HiveMind decomposes it into tasks, retrieves evidence, critiques its own findings, writes a structured report, and evaluates whether the report is good enough — looping autonomously until it is.

```
Query: "What are the risks and mitigations of deploying LLMs in healthcare settings?"

→ Planner     Decomposes into 4 research tasks
              [regulatory risks] [hallucination risks] [HIPAA compliance] [mitigation strategies]

→ Researcher  Retrieves evidence from your documents via ChromaDB + cosine reranking
              confidence: 0.81  sources: 7 chunks across 3 documents

→ Critic      Evaluates findings: "hallucination risks section lacks concrete mitigation detail"
              overall_quality: acceptable  issues: 1

→ Writer      Synthesises plan + findings + critique into a structured report
              sections: Executive Summary, Risk Analysis, Mitigations, Gaps

→ Evaluator   mean retrieval confidence: 0.81 > threshold: 0.50  ✓ SUFFICIENT
              Final report delivered.
```

If confidence is below threshold, the loop continues — automatically generating gap tasks and researching only the missing pieces.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        HIVEMIND SYSTEM                               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     API LAYER (FastAPI)                      │    │
│  │   POST /research/start   GET /research/{id}/stream (SSE)    │    │
│  │   GET  /research/history GET /research/{id}/report          │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                             │                                        │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               ORCHESTRATION LAYER                            │   │
│  │                                                              │   │
│  │   ResearchOrchestrator (sequential pipeline)                 │   │
│  │                                                              │   │
│  │   PLANNING ──▶ RESEARCHING ──▶ CRITIQUING ──▶ REPORTING     │   │
│  │                     ▲                              │         │   │
│  │                     │        EVALUATING ◀──────────┘         │   │
│  │                     │             │                          │   │
│  │                     └─ gap tasks ─┘  (loop until confident)  │   │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                             │                                        │
│              ┌──────────────┼──────────────┐                        │
│              ▼              ▼              ▼                        │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────────────┐    │
│  │  AGENT LAYER  │  │  MEMORY/RAG   │  │  INFRASTRUCTURE      │    │
│  │               │  │               │  │                      │    │
│  │  • Planner    │  │  ChromaDB     │  │  Redis (task queue)  │    │
│  │  • Researcher │◀─│  + cosine     │  │  SQLite (sessions)   │    │
│  │  • Critic     │  │    reranker   │  │  OpenTelemetry       │    │
│  │  • Writer     │  │               │  │  Docker Compose      │    │
│  │  • Evaluator  │  │  Token budget │  │                      │    │
│  └───────────────┘  │  management   │  └──────────────────────┘    │
│                     └───────────────┘                               │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              LLM EXECUTION LAYER                             │   │
│  │   LLMExecutionManager (asyncio.Lock — one call at a time)    │   │
│  │   Providers: Groq (cloud) / Ollama (local) — swappable       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ How It Works

### Step-by-step

**1. Submit a query**
POST to `/research/start`. A session ID is created and the pipeline begins.

**2. Planning** *(runs once)*
The PlannerAgent decomposes the query into a structured list of research tasks with priorities. Problem decomposition is a static step — the original query doesn't change, so planning only runs once regardless of how many iterations follow.

**3. Research loop**
Each task goes to the ResearchAgent, which retrieves relevant document chunks from ChromaDB, reranks them by cosine similarity, and synthesises findings. Confidence is calculated as the mean retrieval score — mathematically, not by asking the LLM to guess.

**4. Critique**
The CriticAgent reviews all findings for completeness, contradictions, and quality. Returns structured feedback: issues, suggestions, and an overall rating (`poor` / `acceptable` / `good`).

**5. Report writing**
The ReportWriterAgent synthesises the original plan, all findings, and the critique into a structured final report with named sections and an executive summary.

**6. Evaluation**
The EvaluatorAgent checks whether the mean confidence score exceeds the configured threshold. If yes: deliver the report. If no: identify the gaps, generate new research tasks for only the missing pieces, and loop back to step 3.

**7. Delivery**
The final report is stored in SQLite and available via REST or SSE stream.

---

## 🔑 Key Engineering Decisions

### Sequential execution over parallel agents
All LLM calls are serialised through a single `asyncio.Lock` inside the `LLMExecutionManager`. This prevents memory exhaustion on local hardware (Ollama) and eliminates rate-limit bursts on free-tier APIs (Groq). Every agent queues and waits — no call is rejected, no concurrency is permitted.

**Tradeoff:** slower wall-clock time per session vs. stable execution on constrained hardware.

---

### Deterministic confidence scoring
Confidence is the mean of vector retrieval scores — a mathematical calculation. The LLM is never asked to self-report certainty.

**Why:** LLMs are overconfident by default. Asking a model "how confident are you?" produces sycophantic, unreliable numbers. Retrieval score is objective.

**Tradeoff:** the score reflects semantic retrieval quality, not the quality of the reasoning in the synthesised report.

---

### ChromaDB over FAISS
ChromaDB provides persistent storage and metadata filtering out of the box. FAISS requires manual index management and has no native persistence.

**Tradeoff:** slightly lower raw ANN search throughput at extreme scale vs. zero-config persistence for this use case.

---

### SSE over WebSockets
Research sessions are one-directional broadcasts — the system streams state updates to the client. SSE is purpose-built for this and requires no connection overhead.

**Tradeoff:** the client cannot send interactive feedback or interrupt mid-stream over the same connection.

---

### Redis for task queue
Redis provides cross-process, persistent task queuing. An `asyncio.Queue` lives in a single event loop's memory — it vanishes on crash and cannot be consumed by multiple workers.

**Tradeoff:** adds an infrastructure dependency for a single use case, increasing operational complexity.

---

### Graceful failure design per agent

| Agent | On failure |
|---|---|
| Planner | Raises `PlannerException` → session fails immediately |
| Researcher | Returns degraded result (`confidence=0.0`) → pipeline continues |
| Critic | Returns auto-approved fallback → pipeline continues |
| Writer | Raises `ReportWriterException` → session fails |
| Evaluator | Returns `sufficient=True` → loop breaks, report delivered |

Critical agents fail hard. Supporting agents fail safe. The system never hangs.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Groq API key (free tier sufficient) **or** Ollama installed locally

### Local development
```bash
git clone https://github.com/TryingtobeingNikhil/HiveMind.git
cd backend
cp .env.example .env
# Add GROQ_API_KEY to .env
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Full stack (recommended)
```bash
docker compose up --build
```

This starts:
- `api` — FastAPI backend on port 8000
- `redis` — task queue broker
- `jaeger` — OpenTelemetry UI on port 16686

### Run tests
```bash
cd backend && python -m pytest tests/ -v
```

---

## 📡 API Reference

| Method | Path | Description | Response |
|---|---|---|---|
| `POST` | `/api/v1/research/start` | Start a research session | `202 Accepted` + session metadata |
| `GET` | `/api/v1/research/history` | List past sessions (paginated) | `200 OK` |
| `GET` | `/api/v1/research/{session_id}` | Full `WorkflowState` | `200 OK` |
| `GET` | `/api/v1/research/{session_id}/status` | Lightweight status + stage | `200 OK` |
| `GET` | `/api/v1/research/{session_id}/report` | Final report | `200 OK` or `404` |
| `GET` | `/api/v1/research/{session_id}/stream` | SSE stream of state updates | Event stream |
| `GET` | `/health` | Liveness check | `{"status": "healthy"}` |
| `GET` | `/ready` | Readiness check | `{"status": "ready", ...}` |

### Example: start a session
```bash
curl -X POST http://localhost:8000/api/v1/research/start \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key risks of deploying LLMs in production?"}'
```
```json
{
  "session_id": "a3f1c9d2-...",
  "status": "queued",
  "created_at": "2026-05-14T10:22:00Z"
}
```

### Example: stream progress
```bash
curl http://localhost:8000/api/v1/research/a3f1c9d2-.../stream
```
```
data: {"stage": "PLANNING", "status": "running", ...}
data: {"stage": "RESEARCHING", "status": "running", "tasks_complete": 2, "tasks_total": 4, ...}
data: {"stage": "EVALUATING", "confidence": 0.81, "sufficient": true, ...}
data: {"stage": "COMPLETED", "status": "completed", ...}
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` for local inference, `groq` for cloud |
| `GROQ_API_KEY` | `None` | Required when `LLM_PROVIDER=groq` |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `MAX_RESEARCH_ITERATIONS` | `2` | Hard cap on research loop iterations |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum mean retrieval score to accept report |
| `EVALUATOR_ENABLED` | `True` | Set `False` to skip evaluation loop entirely |
| `SKIP_CRITIC_FOR_LOW_COMPLEXITY` | `True` | Skip critique stage for 1–2 task plans |
| `OTEL_ENABLED` | `False` | Enable OpenTelemetry distributed tracing |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |

---

## 📁 Project Structure

```
app/
├── agents/        # Planner, Researcher, Critic, Writer, Evaluator
├── api/           # REST endpoints, routing, SSE streaming
├── core/          # Config, exceptions, logging, telemetry
├── db/            # Async SQLite connections, repositories, migrations
├── embedding/     # Vector embedding providers (Ollama / external)
├── llm/           # LLMExecutionManager, rate limiting, provider clients
├── memory/        # Document retrieval, context building, token budgets
├── orchestration/ # Sequential pipeline coordinator + evaluation loop
├── queue/         # Redis-based persistent task queue
├── reranking/     # Cosine similarity reranking of retrieved chunks
├── schemas/       # Pydantic models for all agent I/O
└── vectorstore/   # ChromaDB interaction layer
```

---

## 🔭 Observability

When `OTEL_ENABLED=True`, HiveMind emits OpenTelemetry spans for every agent execution, LLM call, and retrieval operation. Jaeger is included in the Docker Compose stack.

```bash
open http://localhost:16686  # Jaeger UI
```

This gives you a hierarchical waterfall of exactly which agent called what, at what latency — including lock wait time inside `LLMExecutionManager`.

---

## 🗺️ What I'd Do Differently

- **Parallel agent execution** — the sequential lock is a hardware constraint, not a design preference. On adequate hardware, Researcher tasks should fan out concurrently.
- **Authentication** — no auth exists currently; required before any public deployment.
- **WebSockets** — SSE works but doesn't allow mid-session interaction; bidirectional streaming would enable query refinement while research is running.
- **FAISS at scale** — ChromaDB is the right call under ~100K chunks; beyond that, raw FAISS throughput wins.
- **Async task consumer** — `/research/start` currently blocks the HTTP request for the full pipeline duration. A proper background consumer would eliminate gateway timeout risk on long sessions.

---

<div align="center">

**Built by [Nikhil Mourya](https://github.com/TryingtobeingNikhil)** · May 2026

*HiveMind is what RAG looks like when you stop pretending one retrieval pass is enough.*

</div>
