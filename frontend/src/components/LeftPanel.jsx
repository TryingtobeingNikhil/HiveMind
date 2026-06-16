import { useState } from 'react'
import QueryComposer from './LeftPanel/QueryComposer'
import SessionHistory from './LeftPanel/SessionHistory'

/**
 * Left panel — 280px fixed width.
 * Props:
 *   activeSession  — WorkflowState | null
 *   history        — Array<SessionSummary>
 *   isRunning      — bool
 *   onSubmit       — (query: string) => void
 *   onLoad         — (session_id: string) => void
 */
export default function LeftPanel({ activeSession, history, isRunning, onSubmit, onLoad }) {
  const [query, setQuery] = useState('')

  const handleSubmit = () => {
    const trimmed = query.trim()
    if (!trimmed || isRunning) return
    onSubmit(trimmed)
    setQuery('')
  }

  const complexity = activeSession?.plan?.estimated_complexity ?? null

  return (
    <aside
      className="flex flex-col shrink-0 h-full"
      style={{
        width: '280px',
        borderRight: '1px solid var(--border-subtle)',
        backgroundColor: 'var(--bg-surface)',
      }}
    >
      {/* Query composer — fixed top section */}
      <QueryComposer
        query={query}
        setQuery={setQuery}
        isRunning={isRunning}
        onSubmit={handleSubmit}
        complexity={complexity}
      />

      {/* Divider */}
      <div style={{ height: '1px', backgroundColor: 'var(--border-subtle)', flexShrink: 0 }} />

      {/* Session history — fills remaining space, scrolls internally */}
      <div className="flex flex-col flex-1 min-h-0 pt-4">
        <SessionHistory
          history={history}
          activeId={activeSession?.session_id ?? null}
          onLoad={onLoad}
        />
      </div>
    </aside>
  )
}
