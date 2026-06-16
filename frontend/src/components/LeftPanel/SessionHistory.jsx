import { timeAgo, shortId } from '../../utils/formatters'
import { stageColor } from '../../utils/formatters'

/**
 * Session history list in the left panel.
 * Props:
 *   history       — Array<SessionSummary>
 *   activeId      — string | null  (currently viewed session_id)
 *   onLoad        — (session_id: string) => void
 */
export default function SessionHistory({ history, activeId, onLoad }) {
  const isEmpty = !history || history.length === 0

  return (
    <div className="flex flex-col min-h-0 flex-1 px-4 pb-4">
      {/* Section label */}
      <p
        className="text-xs font-semibold uppercase tracking-widest mb-3"
        style={{ color: 'var(--text-muted)' }}
      >
        Recent sessions
      </p>

      {isEmpty ? (
        <div
          className="flex-1 flex items-center justify-center"
        >
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            No sessions yet
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-1 panel-scroll flex-1">
          {history.slice(0, 10).map((session) => {
            const isActive = session.session_id === activeId
            return (
              <button
                key={session.session_id}
                onClick={() => onLoad(session.session_id)}
                className="w-full text-left rounded-lg px-3 py-2.5 transition-all duration-150 group focus-ring"
                style={{
                  backgroundColor: isActive ? 'var(--bg-hover)' : 'transparent',
                  border: isActive
                    ? '1px solid var(--accent-dim)'
                    : '1px solid transparent',
                }}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = 'var(--bg-raised)'
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = 'transparent'
                }}
              >
                {/* Query text */}
                <p
                  className="line-clamp-2 text-xs leading-relaxed mb-1.5"
                  style={{ color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)' }}
                >
                  {session.query}
                </p>

                {/* Meta row */}
                <div className="flex items-center gap-2">
                  {/* Stage dot */}
                  <span
                    className="w-1.5 h-1.5 rounded-full shrink-0"
                    style={{
                      backgroundColor: stageColor(
                        session.status === 'completed' ? 'completed' :
                        session.status === 'failed' ? 'failed' :
                        session.current_stage
                      ),
                    }}
                  />
                  {/* Time */}
                  <span
                    className="text-xs"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    {timeAgo(session.created_at)}
                  </span>
                  {/* Short ID */}
                  <span
                    className="ml-auto font-mono text-xs"
                    style={{ color: 'var(--text-muted)', fontSize: '10px' }}
                  >
                    {shortId(session.session_id)}
                  </span>
                </div>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
