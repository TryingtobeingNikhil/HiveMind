// ── api.js ──────────────────────────────────────────────────────────
// All HTTP calls to the backend. The Vite dev server proxies /api →
// http://localhost:8000 so there are no CORS issues.
// ────────────────────────────────────────────────────────────────────

const BASE = '/api/v1'

/**
 * POST /api/v1/research/start
 * @param {string} query
 * @returns {Promise<{ session_id: string, status: string, created_at: string }>}
 */
export async function startResearch(query) {
  const res = await fetch(`${BASE}/research/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Failed to start research: ${res.status} ${text}`)
  }
  return res.json()
}

/**
 * Opens an SSE stream for a session.
 * Each message is a full WorkflowState JSON string — we JSON.parse it here
 * so callers always receive an object, never a raw string.
 *
 * @param {string} sessionId
 * @param {(state: object) => void} onMessage  — called with parsed WorkflowState
 * @param {(err: Event) => void}   onError    — called on EventSource error
 * @returns {() => void} cleanup function — call this to close the EventSource
 */
export function streamResearch(sessionId, onMessage, onError) {
  const url = `${BASE}/research/${sessionId}/stream`
  const es = new EventSource(url)

  es.onmessage = (event) => {
    try {
      // CRITICAL: event.data is a raw string — must parse to object
      const state = JSON.parse(event.data)
      onMessage(state)
    } catch (err) {
      console.error('[SSE] Failed to parse event.data:', event.data, err)
    }
  }

  es.onerror = (err) => {
    onError(err)
  }

  // Return a cleanup function so callers can close the stream
  return () => {
    es.close()
  }
}

/**
 * GET /api/v1/research/history
 * @param {number} limit
 * @param {number} offset
 * @returns {Promise<Array>}
 */
export async function getHistory(limit = 20, offset = 0) {
  const res = await fetch(`${BASE}/research/history?limit=${limit}&offset=${offset}`)
  if (!res.ok) throw new Error(`Failed to fetch history: ${res.status}`)
  return res.json()
}

/**
 * GET /api/v1/research/{session_id}/report
 * @param {string} sessionId
 * @returns {Promise<object>}
 */
export async function getReport(sessionId) {
  const res = await fetch(`${BASE}/research/${sessionId}/report`)
  if (!res.ok) throw new Error(`Failed to fetch report: ${res.status}`)
  return res.json()
}

/**
 * Health check. Tries /api/v1/health then falls back to /api/health.
 * Returns true if backend is reachable.
 * @returns {Promise<boolean>}
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(5000) })
    return res.ok
  } catch {
    try {
      const res = await fetch('/api/health', { signal: AbortSignal.timeout(5000) })
      return res.ok
    } catch {
      return false
    }
  }
}
