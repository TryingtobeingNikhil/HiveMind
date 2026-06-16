import { useState, useEffect, useRef, useCallback } from 'react'
import { startResearch as apiStart, streamResearch, getHistory, getReport } from '../utils/api'

/**
 * useResearch — central state manager for the research command center.
 *
 * Owns:
 *   activeSession  — current WorkflowState or null (from SSE or loaded report)
 *   history        — Array<SessionSummary> loaded from /history
 *   isRunning      — true while an SSE stream is open
 *
 * Exposes:
 *   startResearch(query)    — POST start, open SSE, update state on each event
 *   loadSession(session_id) — GET report, set as activeSession (read-only)
 */
export function useResearch() {
  const [activeSession, setActiveSession] = useState(null)
  const [history, setHistory] = useState([])
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState(null)

  // Store the SSE cleanup function in a ref so the useEffect cleanup
  // always closes the CURRENT EventSource, even after re-renders.
  // This is the key to preventing duplicate SSE events on remount.
  const cleanupSSERef = useRef(null)

  // ── Load history on mount ──────────────────────────────────────────
  useEffect(() => {
    getHistory(10, 0)
      .then(setHistory)
      .catch((err) => console.error('[useResearch] history load failed:', err))
  }, [])

  // ── Cleanup SSE on unmount ─────────────────────────────────────────
  useEffect(() => {
    return () => {
      // If the component unmounts while a stream is open, close it.
      if (cleanupSSERef.current) {
        cleanupSSERef.current()
        cleanupSSERef.current = null
      }
    }
  }, [])

  // ── startResearch ──────────────────────────────────────────────────
  const startResearch = useCallback(async (query) => {
    if (isRunning) return

    // Close any pre-existing stream before opening a new one
    if (cleanupSSERef.current) {
      cleanupSSERef.current()
      cleanupSSERef.current = null
    }

    setError(null)
    setIsRunning(true)
    setActiveSession(null)

    let sessionId
    try {
      const { session_id } = await apiStart(query)
      sessionId = session_id
    } catch (err) {
      console.error('[useResearch] startResearch failed:', err)
      setError(err.message)
      setIsRunning(false)
      return
    }

    // Open SSE stream
    const closeSSE = streamResearch(
      sessionId,

      // onMessage — receives already-parsed WorkflowState object
      (state) => {
        setActiveSession(state)

        // Terminal states: close the stream
        if (state.status === 'completed' || state.status === 'failed') {
          if (cleanupSSERef.current) {
            cleanupSSERef.current()
            cleanupSSERef.current = null
          }
          setIsRunning(false)

          // Update history list with the finished session
          setHistory((prev) => {
            const summary = {
              session_id: state.session_id,
              query: state.query,
              status: state.status,
              current_stage: state.current_stage,
              created_at: state.created_at,
              completed_at: state.completed_at,
            }
            // Replace existing entry if present, otherwise prepend
            const exists = prev.findIndex((s) => s.session_id === state.session_id)
            if (exists !== -1) {
              const next = [...prev]
              next[exists] = summary
              return next
            }
            return [summary, ...prev].slice(0, 10)
          })
        }
      },

      // onError
      (err) => {
        console.error('[useResearch] SSE error:', err)
        // Only treat as fatal if we don't already have a completed session
        setActiveSession((prev) => {
          if (prev && (prev.status === 'completed' || prev.status === 'failed')) return prev
          return prev
        })
        // Don't forcibly set isRunning=false here — the SSE API fires onerror
        // on reconnect attempts too. Let the terminal state handler do it.
      }
    )

    // Store cleanup so we can close it from anywhere
    cleanupSSERef.current = closeSSE
  }, [isRunning])

  // ── loadSession ────────────────────────────────────────────────────
  const loadSession = useCallback(async (sessionId) => {
    // Close any running stream first
    if (cleanupSSERef.current) {
      cleanupSSERef.current()
      cleanupSSERef.current = null
      setIsRunning(false)
    }

    try {
      const report = await getReport(sessionId)
      setActiveSession(report)
    } catch (err) {
      console.error('[useResearch] loadSession failed:', err)
      setError(err.message)
    }
  }, [])

  return {
    activeSession,
    history,
    isRunning,
    error,
    startResearch,
    loadSession,
  }
}
