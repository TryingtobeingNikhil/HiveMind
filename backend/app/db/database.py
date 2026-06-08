"""
app/db/database.py
──────────────────
Async SQLite database connection and schema initialisation.

Uses aiosqlite for fully non-blocking database operations.
The connection is opened during application lifespan startup and
closed on shutdown, shared across all repositories via app.state.

Schema:
    documents              — document metadata + status
    retrieval_metrics      — retrieval observability records
    research_sessions      — Phase 3 research session state
    agent_execution_metrics — Phase 3 per-agent timing records
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# ── DDL statements ────────────────────────────────────────────────────────────

_CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    file_type     TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'uploaded',
    char_count    INTEGER,
    token_count   INTEGER,
    chunk_count   INTEGER,
    created_at    TEXT NOT NULL,
    processed_at  TEXT,
    error_message TEXT
);
"""

_CREATE_RETRIEVAL_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS retrieval_metrics (
    metrics_id            TEXT PRIMARY KEY,
    query                 TEXT NOT NULL,
    timestamp             TEXT NOT NULL,
    retrieved_chunk_ids   TEXT NOT NULL,
    retrieval_scores      TEXT NOT NULL,
    rerank_scores         TEXT,
    retrieval_latency_ms  REAL NOT NULL,
    total_results         INTEGER NOT NULL
);
"""

_CREATE_DOCUMENTS_IDX = """
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);
"""

_CREATE_METRICS_IDX = """
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON retrieval_metrics (timestamp);
"""

# ── Phase 3: Research session tables ─────────────────────────────────────────

_CREATE_RESEARCH_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS research_sessions (
    session_id      TEXT PRIMARY KEY,
    query           TEXT NOT NULL,
    status          TEXT NOT NULL,
    current_stage   TEXT NOT NULL,
    plan_json       TEXT,
    results_json    TEXT,
    critic_json     TEXT,
    report_json     TEXT,
    error           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    completed_at    TEXT,
    iteration        INTEGER NOT NULL DEFAULT 1,
    evaluations_json TEXT
);
"""

_CREATE_AGENT_EXECUTION_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_execution_metrics (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    agent_name      TEXT NOT NULL,
    stage           TEXT NOT NULL,
    execution_time  REAL NOT NULL,
    model_used      TEXT,
    started_at      TEXT NOT NULL,
    completed_at    TEXT NOT NULL
);
"""

_CREATE_RESEARCH_SESSIONS_IDX = """
CREATE INDEX IF NOT EXISTS idx_research_sessions_status
    ON research_sessions (status);
"""

_CREATE_AGENT_METRICS_IDX = """
CREATE INDEX IF NOT EXISTS idx_agent_metrics_session
    ON agent_execution_metrics (session_id);
"""

# ── Database initialisation ───────────────────────────────────────────────────


async def init_database(db_path: Path) -> aiosqlite.Connection:
    """
    Open an aiosqlite connection and ensure all tables exist.

    Args:
        db_path: Absolute path to the SQLite database file.
                 Parent directory is created if it does not exist.

    Returns:
        An open :class:`aiosqlite.Connection` ready for use.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising database", extra={"path": str(db_path)})

    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row  # rows accessible as dicts

    async with conn.executescript(
        "\n".join([
            _CREATE_DOCUMENTS_TABLE,
            _CREATE_RETRIEVAL_METRICS_TABLE,
            _CREATE_DOCUMENTS_IDX,
            _CREATE_METRICS_IDX,
            # Phase 3 tables
            _CREATE_RESEARCH_SESSIONS_TABLE,
            _CREATE_AGENT_EXECUTION_METRICS_TABLE,
            _CREATE_RESEARCH_SESSIONS_IDX,
            _CREATE_AGENT_METRICS_IDX,
        ])
    ):
        pass

    await conn.commit()

    # ── Phase 4: migration — add new columns if they don't exist yet ──────────
    # Using ALTER TABLE with try/except because SQLite raises OperationalError
    # if the column already exists. This is safe to run on every startup.
    _phase4_migrations = [
        "ALTER TABLE research_sessions ADD COLUMN iteration INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE research_sessions ADD COLUMN evaluations_json TEXT",
    ]
    for stmt in _phase4_migrations:
        try:
            await conn.execute(stmt)
            await conn.commit()
        except Exception:
            pass  # Column already exists — safe to ignore

    logger.info("Database initialised", extra={"path": str(db_path)})
    return conn


async def close_database(conn: aiosqlite.Connection) -> None:
    """Close the database connection gracefully."""
    await conn.close()
    logger.info("Database connection closed")
