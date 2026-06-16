/**
 * ResearchPlan — shows the planned tasks with completion tracking.
 * Props:
 *   plan     — { tasks: Array<{ task_id, query, priority }>, estimated_complexity }
 *   results  — Array<{ task_id, ... }>  (used to mark tasks complete)
 */
export default function ResearchPlan({ plan, results }) {
  if (!plan || !plan.tasks?.length) return null

  const completedIds = new Set((results ?? []).map((r) => r.task_id))

  const complexityColors = {
    low:    'bg-green-500/10 text-green-400 border-green-500/20',
    medium: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    high:   'bg-red-500/10 text-red-400 border-red-500/20',
  }

  // Priority 1 = highest. Scale dot opacity: priority 1 → full, higher numbers → dimmer
  const priorityOpacity = (p) => Math.max(0.2, 1 - (p - 1) * 0.2)

  return (
    <div
      className="rounded-xl p-4 animate-fade-in"
      style={{
        backgroundColor: 'var(--bg-raised)',
        border: '1px solid var(--border-subtle)',
      }}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span
          className="text-xs font-semibold"
          style={{ color: 'var(--text-primary)' }}
        >
          Research plan
        </span>
        {plan.estimated_complexity && (
          <span
            className={`text-xs px-2 py-0.5 rounded-full border font-medium ${complexityColors[plan.estimated_complexity] ?? 'bg-white/5 text-[var(--text-muted)] border-white/10'}`}
          >
            {plan.estimated_complexity}
          </span>
        )}
        <span
          className="ml-auto text-xs"
          style={{ color: 'var(--text-muted)' }}
        >
          {completedIds.size}/{plan.tasks.length}
        </span>
      </div>

      {/* Task list */}
      <div className="flex flex-col gap-2">
        {plan.tasks.map((task) => {
          const done = completedIds.has(task.task_id)
          return (
            <div
              key={task.task_id}
              className="flex items-start gap-2.5 py-2 px-2 rounded-lg transition-colors duration-200"
              style={{
                backgroundColor: done ? 'var(--bg-hover)' : 'transparent',
              }}
            >
              {/* Priority dot */}
              <span
                className="mt-1 shrink-0 w-1.5 h-1.5 rounded-full"
                style={{
                  backgroundColor: 'var(--accent)',
                  opacity: priorityOpacity(task.priority ?? 1),
                }}
              />

              {/* Task query */}
              <span
                className="text-xs leading-relaxed flex-1"
                style={{
                  color: done ? 'var(--text-muted)' : 'var(--text-secondary)',
                  textDecoration: done ? 'line-through' : 'none',
                }}
              >
                {task.query}
              </span>

              {/* Completion indicator */}
              {done ? (
                <svg
                  viewBox="0 0 12 12"
                  fill="none"
                  stroke="var(--stage-completed)"
                  strokeWidth={2.5}
                  className="w-3 h-3 shrink-0 mt-0.5 animate-fade-in"
                >
                  <polyline points="2,6 5,9 10,3" />
                </svg>
              ) : (
                <span
                  className="w-3 h-3 rounded shrink-0 mt-0.5"
                  style={{
                    border: '1px solid var(--text-muted)',
                    opacity: 0.4,
                  }}
                />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
