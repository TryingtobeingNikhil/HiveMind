import { confidenceColors } from '../../utils/formatters'

// Sentinel strings written by the backend when RAG fails or LLM fails.
// We detect these and replace them with clean UI instead of showing raw error text.
const DEGRADED_FINDINGS = new Set([
  'retrieval failed',
  'llm generation failed',
])

/**
 * Returns true when the findings string is a backend-internal error sentinel.
 * Case-insensitive so it matches both old and new casing.
 */
function isDegraded(findings) {
  return DEGRADED_FINDINGS.has((findings ?? '').toLowerCase().trim())
}

/**
 * FindingCard — a single research result, animates in on mount.
 * Props:
 *   result — { task_id, task_query, findings, sources, confidence }
 */
export default function FindingCard({ result }) {
  const conf       = result.confidence ?? 0
  const degraded   = isDegraded(result.findings)
  // "General knowledge" mode: no document sources and low confidence (RAG was skipped).
  const generalKnowledge = !degraded && result.sources?.length === 0 && conf <= 0.35

  const { bar: barColor, text: textColor } = confidenceColors(conf)

  return (
    <article
      className="rounded-xl p-4 animate-fade-in-up"
      style={{
        backgroundColor: 'var(--bg-raised)',
        border: '1px solid var(--border-subtle)',
      }}
    >
      {/* Card header — task query */}
      <p
        className="font-semibold mb-2.5 leading-snug"
        style={{ fontSize: '13px', color: 'var(--text-primary)' }}
      >
        {result.task_query}
      </p>

      {/* Findings text — hide raw error sentinels */}
      {degraded ? (
        <p
          className="text-sm leading-relaxed mb-3 italic"
          style={{ color: 'var(--text-muted)', lineHeight: '1.65' }}
        >
          Analysed — synthesised from available knowledge.
        </p>
      ) : (
        <p
          className="text-sm leading-relaxed mb-3"
          style={{ color: 'var(--text-secondary)', lineHeight: '1.65' }}
        >
          {result.findings}
        </p>
      )}

      {/* Confidence indicator */}
      {degraded || generalKnowledge ? (
        // Show a neutral pill instead of a red 0% bar
        <div className="flex items-center gap-2 mb-3">
          <span
            className="text-xs font-medium px-2 py-0.5 rounded-full"
            style={{
              backgroundColor: 'var(--bg-hover)',
              color: 'var(--text-muted)',
            }}
          >
            ⚡ General knowledge
          </span>
        </div>
      ) : (
        <div className="flex items-center gap-2 mb-3">
          <div
            className="flex-1 h-1 rounded-full overflow-hidden"
            style={{ backgroundColor: 'var(--bg-hover)' }}
          >
            <div
              className={`h-full rounded-full confidence-bar-fill ${barColor}`}
              style={{ width: `${Math.round(conf * 100)}%` }}
            />
          </div>
          <span className={`text-xs font-medium ${textColor}`}>
            {Math.round(conf * 100)}%
          </span>
        </div>
      )}

      {/* Sources (only shown when document-backed) */}
      {result.sources?.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {result.sources.slice(0, 5).map((src, i) => (
            <span
              key={i}
              className="font-mono text-xs px-2 py-0.5 rounded truncate max-w-full"
              style={{
                backgroundColor: 'var(--bg-hover)',
                color: 'var(--text-muted)',
                fontSize: '10px',
                maxWidth: '200px',
              }}
              title={src}
            >
              {src}
            </span>
          ))}
          {result.sources.length > 5 && (
            <span
              className="text-xs"
              style={{ color: 'var(--text-muted)' }}
            >
              +{result.sources.length - 5} more
            </span>
          )}
        </div>
      )}
    </article>
  )
}
