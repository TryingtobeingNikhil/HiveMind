import { useHealthCheck } from '../hooks/useHealthCheck'

export default function Header() {
  const { isOnline } = useHealthCheck()

  return (
    <header
      className="flex items-center justify-between px-5 shrink-0"
      style={{
        height: '48px',
        borderBottom: '1px solid var(--border-subtle)',
        backgroundColor: 'var(--bg-surface)',
      }}
    >
      {/* Left — logo */}
      <div className="flex items-center gap-2.5">
        <span
          className="font-mono font-bold text-sm tracking-tight"
          style={{ color: 'var(--accent)' }}
        >
          ODR
        </span>
        <span
          className="text-xs font-medium hidden sm:block"
          style={{ color: 'var(--text-secondary)' }}
        >
          Open Deep Research
        </span>
      </div>

      {/* Right — backend status */}
      <div className="flex items-center gap-2">
        <span
          className="relative flex h-2 w-2"
          aria-label={isOnline ? 'Backend online' : 'Backend offline'}
        >
          {isOnline && (
            <span
              className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
              style={{ backgroundColor: 'var(--stage-completed)' }}
            />
          )}
          <span
            className="relative inline-flex rounded-full h-2 w-2"
            style={{
              backgroundColor: isOnline
                ? 'var(--stage-completed)'
                : 'var(--stage-failed)',
            }}
          />
        </span>
        <span
          className="text-xs"
          style={{ color: 'var(--text-secondary)' }}
        >
          {isOnline ? 'API Connected' : 'API Offline'}
        </span>
      </div>
    </header>
  )
}
