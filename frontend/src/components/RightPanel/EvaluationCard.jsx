/**
 * EvaluationCard — shows one evaluation iteration's result.
 * Props:
 *   evaluation — {
 *     iteration: number,
 *     sufficient: boolean,
 *     confidence_score: number,
 *     confidence_threshold: number,
 *     gaps: string[],
 *     gap_tasks: Array<{ task_id, query }>
 *   }
 */
export default function EvaluationCard({ evaluation }) {
  const {
    iteration,
    sufficient,
    confidence_score = 0,
    confidence_threshold = 0.7,
    gaps = [],
    gap_tasks = [],
  } = evaluation

  const pct  = Math.round(confidence_score * 100)
  const thPct = Math.round(confidence_threshold * 100)

  return (
    <div
      className="rounded-xl p-4 animate-fade-in"
      style={{
        backgroundColor: 'var(--bg-raised)',
        border: '1px solid var(--border-subtle)',
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>
          Iteration {iteration}
        </span>
        {sufficient ? (
          <span className="flex items-center gap-1 text-xs font-medium" style={{ color: 'var(--stage-completed)' }}>
            <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth={2.5} className="w-3 h-3">
              <polyline points="2,6 5,9 10,3" />
            </svg>
            Sufficient
          </span>
        ) : (
          <span className="flex items-center gap-1 text-xs font-medium" style={{ color: 'var(--stage-evaluating)' }}>
            <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth={2} className="w-3 h-3">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 2v8M10 6l-4 4-4-4" />
            </svg>
            Insufficient — continuing
          </span>
        )}
      </div>

      {/* Confidence bar with threshold marker */}
      <div className="mb-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Confidence</span>
          <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
            {pct}% / {thPct}%
          </span>
        </div>
        <div
          className="relative h-2 rounded-full overflow-visible"
          style={{ backgroundColor: 'var(--bg-hover)' }}
        >
          {/* Fill */}
          <div
            className="absolute left-0 top-0 h-full rounded-full confidence-bar-fill"
            style={{
              width: `${pct}%`,
              backgroundColor: sufficient ? 'var(--stage-completed)' : 'var(--stage-evaluating)',
            }}
          />
          {/* Threshold line marker */}
          <div
            className="absolute top-[-3px] w-0.5 h-[14px] rounded-full"
            style={{
              left: `${thPct}%`,
              backgroundColor: 'var(--text-muted)',
              transform: 'translateX(-50%)',
            }}
            title={`Threshold: ${thPct}%`}
          />
        </div>
      </div>

      {/* Gaps */}
      {gaps.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>
            Gaps identified
          </p>
          <ul className="flex flex-col gap-1">
            {gaps.map((gap, i) => (
              <li
                key={i}
                className="text-xs leading-relaxed"
                style={{ color: 'var(--text-secondary)' }}
              >
                · {gap}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Gap tasks */}
      {gap_tasks.length > 0 && (
        <div>
          <p className="text-xs font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>
            New tasks
          </p>
          <div className="flex flex-wrap gap-1.5">
            {gap_tasks.map((task) => (
              <span
                key={task.task_id}
                className="text-xs px-2 py-0.5 rounded-full"
                style={{
                  backgroundColor: 'var(--bg-hover)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                {task.query}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
