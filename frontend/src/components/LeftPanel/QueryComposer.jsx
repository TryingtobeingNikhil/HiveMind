import { useRef, useEffect, useCallback } from 'react'

/**
 * Auto-growing textarea query composer.
 * Props:
 *   query        — string
 *   setQuery     — (s: string) => void
 *   isRunning    — bool
 *   onSubmit     — () => void
 *   complexity   — "low" | "medium" | "high" | null
 */
export default function QueryComposer({ query, setQuery, isRunning, onSubmit, complexity }) {
  const textareaRef = useRef(null)

  // Auto-grow the textarea to fit content, up to 160px
  const resize = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }, [])

  useEffect(() => {
    resize()
  }, [query, resize])

  // Cmd+Enter submits
  const handleKeyDown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      if (!isRunning && query.trim()) onSubmit()
    }
  }

  const complexityColors = {
    low:    'bg-green-500/10 text-green-400 border-green-500/20',
    medium: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    high:   'bg-red-500/10 text-red-400 border-red-500/20',
  }

  return (
    <div className="p-4 flex flex-col gap-3">
      {/* Textarea */}
      <div
        className="rounded-lg p-0.5"
        style={{
          background: isRunning
            ? 'linear-gradient(135deg, var(--accent) 0%, transparent 60%)'
            : 'var(--border-subtle)',
        }}
      >
        <div
          className="rounded-[7px] p-3"
          style={{ backgroundColor: 'var(--bg-raised)' }}
        >
          <textarea
            ref={textareaRef}
            id="query-input"
            value={query}
            onChange={(e) => { setQuery(e.target.value); resize() }}
            onKeyDown={handleKeyDown}
            onInput={resize}
            placeholder="What do you want to research?"
            disabled={isRunning}
            rows={3}
            className="query-textarea w-full bg-transparent border-none outline-none text-sm leading-relaxed focus-ring"
            style={{
              color: 'var(--text-primary)',
              minHeight: '80px',
            }}
          />
        </div>
      </div>

      {/* Complexity badge — shown when prior plan has been received */}
      {complexity && (
        <div className="flex items-center gap-1.5 animate-fade-in">
          <span
            className="text-xs"
            style={{ color: 'var(--text-muted)' }}
          >
            Estimated complexity:
          </span>
          <span
            className={`text-xs font-medium px-2 py-0.5 rounded-full border ${complexityColors[complexity] ?? 'bg-white/5 text-[var(--text-muted)] border-white/10'}`}
          >
            {complexity}
          </span>
        </div>
      )}

      {/* Submit button */}
      <button
        id="submit-research-btn"
        onClick={onSubmit}
        disabled={isRunning || !query.trim()}
        className="w-full py-2.5 rounded-lg text-sm font-medium transition-all duration-200 focus-ring relative overflow-hidden"
        style={{
          backgroundColor: isRunning || !query.trim() ? 'var(--bg-hover)' : 'var(--accent)',
          color: isRunning || !query.trim() ? 'var(--text-muted)' : '#fff',
          cursor: isRunning || !query.trim() ? 'not-allowed' : 'pointer',
        }}
      >
        {isRunning ? (
          <span className="flex items-center justify-center gap-2">
            <svg
              className="animate-spin-slow h-3.5 w-3.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
            Researching...
          </span>
        ) : (
          <span className="flex items-center justify-center gap-1.5">
            Research
            <span
              className="text-xs opacity-50"
              style={{ fontSize: '10px' }}
            >
              ⌘↵
            </span>
          </span>
        )}
      </button>
    </div>
  )
}
