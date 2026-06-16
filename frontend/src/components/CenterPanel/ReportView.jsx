import { confidenceColors, reportToMarkdown } from '../../utils/formatters'

/**
 * ReportView — full rendered report when status = completed.
 * Props:
 *   session — full WorkflowState (has .report and .query)
 */
export default function ReportView({ session }) {
  const { report, query } = session
  if (!report) return null

  const conf = report.confidence_score ?? 0
  const { bar: barColor, text: textColor } = confidenceColors(conf)

  async function handleCopy() {
    const md = reportToMarkdown(session)
    try {
      await navigator.clipboard.writeText(md)
    } catch {
      // Fallback for environments where clipboard API is blocked
      const el = document.createElement('textarea')
      el.value = md
      document.body.appendChild(el)
      el.select()
      document.execCommand('copy')
      document.body.removeChild(el)
    }
  }

  return (
    <div className="flex flex-col gap-5 animate-fade-in report-body">
      {/* Title row */}
      <div className="flex items-start justify-between gap-3">
        <h1>{query}</h1>
        <button
          id="copy-report-btn"
          onClick={handleCopy}
          className="shrink-0 flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-colors duration-150 focus-ring"
          style={{
            backgroundColor: 'var(--bg-raised)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border-subtle)',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-hover)'
            e.currentTarget.style.color = 'var(--text-primary)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-raised)'
            e.currentTarget.style.color = 'var(--text-secondary)'
          }}
        >
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-3.5 h-3.5">
            <rect x="5" y="5" width="9" height="9" rx="1.5" />
            <path d="M3 11H2a1 1 0 01-1-1V2a1 1 0 011-1h8a1 1 0 011 1v1" />
          </svg>
          Copy as Markdown
        </button>
      </div>

      {/* Confidence bar */}
      <div
        className="rounded-xl p-4"
        style={{ backgroundColor: 'var(--bg-raised)', border: '1px solid var(--border-subtle)' }}
      >
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
            Confidence score
          </span>
          <span className={`text-xs font-semibold ${textColor}`}>
            {Math.round(conf * 100)}%
          </span>
        </div>
        <div
          className="h-1.5 rounded-full overflow-hidden"
          style={{ backgroundColor: 'var(--bg-hover)' }}
        >
          <div
            className={`h-full rounded-full confidence-bar-fill ${barColor}`}
            style={{ width: `${Math.round(conf * 100)}%` }}
          />
        </div>
        {report.critic_quality && (
          <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
            Critic quality: {report.critic_quality}
          </p>
        )}
      </div>

      {/* Summary */}
      {report.summary && (
        <div
          className="rounded-xl p-4"
          style={{ backgroundColor: 'var(--bg-raised)', border: '1px solid var(--border-subtle)' }}
        >
          <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
            {report.summary}
          </p>
        </div>
      )}

      {/* Sections */}
      {(report.sections ?? []).map((section, i) => (
        <div key={i}>
          <h2>{section.title}</h2>
          <p>{section.content}</p>
        </div>
      ))}

      {/* Citations */}
      {report.citations?.length > 0 && (
        <div>
          <h2>Citations</h2>
          <div className="flex flex-wrap gap-2 mt-2">
            {report.citations.map((cite, i) => (
              <span
                key={i}
                className="text-xs font-mono px-2 py-1 rounded"
                style={{
                  backgroundColor: 'var(--bg-raised)',
                  color: 'var(--text-muted)',
                  border: '1px solid var(--border-subtle)',
                  maxWidth: '100%',
                  overflowWrap: 'break-word',
                }}
              >
                {cite}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
