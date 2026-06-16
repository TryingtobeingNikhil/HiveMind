import FindingCard from './FindingCard'

/**
 * FindingsStream — scrollable list of research results, newest first.
 * Props:
 *   results — Array<ResearchResult>
 */
export default function FindingsStream({ results }) {
  // Reverse so newest results appear at the top
  const reversed = results ? [...results].reverse() : []

  if (reversed.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full px-4">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center mb-3"
          style={{ backgroundColor: 'var(--bg-raised)' }}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.5}
            className="w-5 h-5"
            style={{ color: 'var(--text-muted)' }}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
            />
          </svg>
        </div>
        <p className="text-xs text-center" style={{ color: 'var(--text-muted)' }}>
          Findings will appear here<br />as agents complete
        </p>
      </div>
    )
  }

  return (
    <div className="panel-scroll flex-1 p-3 flex flex-col gap-3">
      {reversed.map((result) => (
        <FindingCard key={result.task_id} result={result} />
      ))}
    </div>
  )
}
