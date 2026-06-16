// ── formatters.js ────────────────────────────────────────────────────
// Pure utility functions for display formatting.
// ────────────────────────────────────────────────────────────────────

/**
 * Returns a human-readable "time ago" string.
 * @param {string|Date} dateInput
 * @returns {string} e.g. "just now", "3 minutes ago", "2 hours ago"
 */
export function timeAgo(dateInput) {
  if (!dateInput) return ''
  const date = typeof dateInput === 'string' ? new Date(dateInput) : dateInput
  const diffMs = Date.now() - date.getTime()
  const diffSec = Math.floor(diffMs / 1000)

  if (diffSec < 10) return 'just now'
  if (diffSec < 60) return `${diffSec} seconds ago`

  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin} minute${diffMin === 1 ? '' : 's'} ago`

  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24) return `${diffHr} hour${diffHr === 1 ? '' : 's'} ago`

  const diffDay = Math.floor(diffHr / 24)
  return `${diffDay} day${diffDay === 1 ? '' : 's'} ago`
}

/**
 * Formats elapsed seconds into a human-readable duration.
 * @param {number} totalSeconds
 * @returns {string} e.g. "1m 23s", "45s"
 */
export function formatElapsed(totalSeconds) {
  if (totalSeconds < 60) return `${totalSeconds}s`
  const m = Math.floor(totalSeconds / 60)
  const s = totalSeconds % 60
  return s === 0 ? `${m}m` : `${m}m ${s}s`
}

/**
 * Truncates a string to maxLen characters with ellipsis.
 * @param {string} str
 * @param {number} maxLen
 * @returns {string}
 */
export function truncate(str, maxLen = 60) {
  if (!str) return ''
  return str.length <= maxLen ? str : str.slice(0, maxLen - 1) + '…'
}

/**
 * Returns Tailwind color class names for a confidence score.
 * @param {number} score  0–1
 * @returns {{ bg: string, text: string, bar: string }}
 */
export function confidenceColors(score) {
  if (score >= 0.7) return { bg: 'bg-green-500/10', text: 'text-green-400', bar: 'bg-green-500' }
  if (score >= 0.4) return { bg: 'bg-amber-500/10', text: 'text-amber-400', bar: 'bg-amber-500' }
  return { bg: 'bg-red-500/10', text: 'text-red-400', bar: 'bg-red-500' }
}

/**
 * Maps a pipeline stage name to its CSS variable color string.
 * @param {string} stage
 * @returns {string} CSS color value
 */
export function stageColor(stage) {
  const map = {
    planning:    'var(--stage-planning)',
    researching: 'var(--stage-researching)',
    critiquing:  'var(--stage-critiquing)',
    reporting:   'var(--stage-reporting)',
    evaluating:  'var(--stage-evaluating)',
    completed:   'var(--stage-completed)',
    failed:      'var(--stage-failed)',
  }
  return map[stage] || 'var(--text-muted)'
}

/**
 * Maps stage name to its Tailwind background color class for badges.
 * @param {string} stage
 * @returns {string}
 */
export function stageBadgeClass(stage) {
  const map = {
    planning:    'bg-amber-500/15 text-amber-400',
    researching: 'bg-blue-500/15 text-blue-400',
    critiquing:  'bg-purple-500/15 text-purple-400',
    reporting:   'bg-emerald-500/15 text-emerald-400',
    evaluating:  'bg-orange-500/15 text-orange-400',
    completed:   'bg-green-500/15 text-green-400',
    failed:      'bg-red-500/15 text-red-400',
    running:     'bg-blue-500/15 text-blue-400',
  }
  return map[stage] || 'bg-white/5 text-[var(--text-muted)]'
}

/**
 * Short-formats a session_id for display (first 8 chars).
 * @param {string} sessionId
 * @returns {string}
 */
export function shortId(sessionId) {
  if (!sessionId) return ''
  return sessionId.slice(0, 8)
}

/**
 * Converts a full WorkflowState into a minimal clipboard-markdown report.
 * @param {object} state - WorkflowState
 * @returns {string} markdown
 */
export function reportToMarkdown(state) {
  if (!state?.report) return ''
  const { report, query } = state
  const lines = []

  lines.push(`# ${query}`)
  lines.push('')
  lines.push(`> **Confidence:** ${Math.round((report.confidence_score ?? 0) * 100)}%  ·  **Quality:** ${report.critic_quality ?? 'N/A'}`)
  lines.push('')

  if (report.summary) {
    lines.push('## Summary')
    lines.push(report.summary)
    lines.push('')
  }

  for (const section of (report.sections ?? [])) {
    lines.push(`## ${section.title}`)
    lines.push(section.content)
    lines.push('')
  }

  if (report.citations?.length) {
    lines.push('## Citations')
    report.citations.forEach((c, i) => lines.push(`${i + 1}. ${c}`))
    lines.push('')
  }

  lines.push(`---`)
  lines.push(`*Generated at ${report.generated_at ? new Date(report.generated_at).toLocaleString() : 'N/A'}*`)

  return lines.join('\n')
}
