/**
 * CriticCard — displays the critic result (quality, issues, suggestions).
 * Props:
 *   criticResult — {
 *     approved: boolean,
 *     issues: string[],
 *     suggestions: string[],
 *     overall_quality: "poor" | "acceptable" | "good"
 *   }
 */
export default function CriticCard({ criticResult }) {
  if (!criticResult) return null

  const { approved, issues = [], suggestions = [], overall_quality } = criticResult

  // Detect auto-approved: approved=true AND no issues AND no suggestions
  const isAutoApproved = approved && issues.length === 0 && suggestions.length === 0

  const qualityConfig = {
    good:       { label: 'Good',       classes: 'bg-green-500/10 text-green-400 border-green-500/20' },
    acceptable: { label: 'Acceptable', classes: 'bg-amber-500/10 text-amber-400 border-amber-500/20' },
    poor:       { label: 'Poor',       classes: 'bg-red-500/10 text-red-400 border-red-500/20' },
  }
  const quality = qualityConfig[overall_quality] ?? {
    label: overall_quality ?? 'Unknown',
    classes: 'bg-white/5 text-[var(--text-muted)] border-white/10',
  }

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
          style={{ color: 'var(--text-secondary)' }}
        >
          Critic review
        </span>
        <span
          className={`text-xs font-medium px-2 py-0.5 rounded-full border ${quality.classes}`}
        >
          {quality.label}
        </span>
        {isAutoApproved && (
          <span
            className="text-xs px-2 py-0.5 rounded-full"
            style={{ backgroundColor: 'var(--bg-hover)', color: 'var(--text-muted)' }}
          >
            Auto-approved
          </span>
        )}
      </div>

      {/* Issues */}
      {issues.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>
            Issues
          </p>
          <ul className="flex flex-col gap-1.5">
            {issues.map((issue, i) => (
              <li key={i} className="flex items-start gap-2">
                <span
                  className="mt-1.5 w-1.5 h-1.5 rounded-full shrink-0"
                  style={{ backgroundColor: 'var(--stage-failed)' }}
                />
                <span className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                  {issue}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <div>
          <p className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>
            Suggestions
          </p>
          <ul className="flex flex-col gap-1.5">
            {suggestions.map((suggestion, i) => (
              <li key={i} className="flex items-start gap-2">
                <span
                  className="mt-1.5 w-1.5 h-1.5 rounded-full shrink-0"
                  style={{ backgroundColor: 'var(--stage-researching)' }}
                />
                <span className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                  {suggestion}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Auto-approved message */}
      {isAutoApproved && (
        <p className="text-xs mt-2" style={{ color: 'var(--text-muted)' }}>
          Research quality was sufficient — critic review was skipped automatically.
        </p>
      )}
    </div>
  )
}
