import { useState, useEffect } from 'react'
import { shortId, formatElapsed, stageBadgeClass } from '../utils/formatters'
import PipelineTrack from './CenterPanel/PipelineTrack'
import ResearchPlan from './CenterPanel/ResearchPlan'
import ReportView from './CenterPanel/ReportView'
import EmptyState from './CenterPanel/EmptyState'

/**
 * CenterPanel — the main content area.
 * Props:
 *   session   — WorkflowState | null
 *   isRunning — bool
 */
export default function CenterPanel({ session, isRunning }) {
  // Live elapsed timer — counts up while session is running
  const [elapsedSec, setElapsedSec] = useState(0)

  useEffect(() => {
    if (!session || !isRunning) {
      setElapsedSec(0)
      return
    }

    const start = session.created_at ? new Date(session.created_at).getTime() : Date.now()

    const tick = () => {
      setElapsedSec(Math.floor((Date.now() - start) / 1000))
    }
    tick()
    const id = setInterval(tick, 1000)

    return () => clearInterval(id)
  }, [session?.session_id, isRunning])

  if (!session) {
    return (
      <main
        className="flex-1 min-w-0 h-full"
        style={{ backgroundColor: 'var(--bg-base)' }}
      >
        <EmptyState />
      </main>
    )
  }

  const { status, current_stage, iteration, plan, results = [], report, error: sessionError } = session

  const showReport = status === 'completed' && report
  const showError  = status === 'failed'

  return (
    <main
      className="flex-1 min-w-0 h-full flex flex-col overflow-hidden"
      style={{ backgroundColor: 'var(--bg-base)' }}
    >
      {/* Session header */}
      <div
        className="px-6 py-4 shrink-0"
        style={{ borderBottom: '1px solid var(--border-subtle)' }}
      >
        <div className="flex items-start gap-3 flex-wrap">
          {/* Query */}
          <p
            className="text-sm font-medium flex-1 leading-relaxed"
            style={{ color: 'var(--text-primary)', minWidth: '200px' }}
          >
            {session.query}
          </p>

          {/* Badges */}
          <div className="flex items-center gap-2 shrink-0">
            {/* Iteration badge */}
            {iteration > 1 && (
              <span
                className="text-xs px-2 py-0.5 rounded-full font-medium"
                style={{
                  backgroundColor: 'var(--accent-dim)',
                  color: 'var(--accent)',
                }}
              >
                Iteration {iteration}
              </span>
            )}

            {/* Status badge */}
            <span
              className={`text-xs px-2 py-0.5 rounded-full font-medium ${stageBadgeClass(status === 'running' ? current_stage : status)}`}
            >
              {status === 'running' ? current_stage : status}
            </span>

            {/* Elapsed timer */}
            {isRunning && (
              <span
                className="font-mono text-xs"
                style={{ color: 'var(--text-muted)' }}
              >
                {formatElapsed(elapsedSec)}
              </span>
            )}

            {/* Short session ID */}
            <span
              className="font-mono text-xs"
              style={{ color: 'var(--text-muted)', fontSize: '10px' }}
            >
              {shortId(session.session_id)}
            </span>
          </div>
        </div>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 panel-scroll px-6 py-5 flex flex-col gap-6">
        {/* Pipeline progress track */}
        <div
          className="rounded-xl p-4"
          style={{
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <p
            className="text-xs font-medium mb-4"
            style={{ color: 'var(--text-muted)' }}
          >
            Pipeline
          </p>
          <PipelineTrack currentStage={current_stage} status={status} />
        </div>

        {/* Research plan */}
        {plan && (
          <ResearchPlan plan={plan} results={results} />
        )}

        {/* Error state */}
        {showError && (
          <div
            className="rounded-xl p-5 animate-fade-in"
            style={{
              backgroundColor: 'rgba(239,68,68,0.05)',
              border: '1px solid rgba(239,68,68,0.25)',
            }}
          >
            <div className="flex items-center gap-2 mb-2">
              <span style={{ color: 'var(--stage-failed)' }}>✕</span>
              <p
                className="text-sm font-semibold"
                style={{ color: 'var(--stage-failed)' }}
              >
                Research failed
              </p>
            </div>
            {sessionError && (
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                {sessionError}
              </p>
            )}
          </div>
        )}

        {/* Report */}
        {showReport && <ReportView session={session} />}
      </div>
    </main>
  )
}
