/**
 * Empty state shown in the center panel when no session is active.
 */
export default function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4 select-none">
      <div
        className="w-14 h-14 rounded-2xl flex items-center justify-center"
        style={{ backgroundColor: 'var(--bg-raised)' }}
      >
        <span className="text-2xl" style={{ color: 'var(--text-muted)' }}>→</span>
      </div>
      <div className="text-center">
        <p
          className="text-sm font-medium mb-1"
          style={{ color: 'var(--text-secondary)' }}
        >
          Submit a query to begin
        </p>
        <p
          className="text-xs"
          style={{ color: 'var(--text-muted)' }}
        >
          Results will stream in real time
        </p>
      </div>
    </div>
  )
}
