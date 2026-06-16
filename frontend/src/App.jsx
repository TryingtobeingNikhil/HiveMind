import { useState } from 'react'
import Header from './components/Header'
import LeftPanel from './components/LeftPanel'
import CenterPanel from './components/CenterPanel'
import RightPanel from './components/RightPanel'
import { useResearch } from './hooks/useResearch'

/**
 * App — three-panel command center layout.
 *
 * Desktop (≥1024px): fixed-width left (280px) + flex center + fixed-width right (320px)
 * Tablet / mobile (<1024px): single column, left panel becomes a bottom sheet trigger.
 */
export default function App() {
  const { activeSession, history, isRunning, startResearch, loadSession } = useResearch()
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false)

  return (
    <div
      className="flex flex-col"
      style={{ height: '100dvh', backgroundColor: 'var(--bg-base)' }}
    >
      {/* Header */}
      <Header />

      {/* Three-panel body */}
      <div className="flex flex-1 min-h-0 relative">

        {/* ── Desktop: Left panel ── */}
        <div className="hidden lg:flex">
          <LeftPanel
            activeSession={activeSession}
            history={history}
            isRunning={isRunning}
            onSubmit={startResearch}
            onLoad={(id) => {
              loadSession(id)
              setMobileDrawerOpen(false)
            }}
          />
        </div>

        {/* ── Center panel ── */}
        <CenterPanel session={activeSession} isRunning={isRunning} />

        {/* ── Desktop: Right panel ── */}
        <div className="hidden lg:flex">
          <RightPanel session={activeSession} />
        </div>

        {/* ── Mobile: bottom sheet overlay ── */}
        {mobileDrawerOpen && (
          <div
            className="fixed inset-0 z-40 lg:hidden"
            onClick={() => setMobileDrawerOpen(false)}
            style={{ backgroundColor: 'rgba(0,0,0,0.6)' }}
          />
        )}
        <div
          className={`fixed bottom-0 left-0 right-0 z-50 lg:hidden transition-transform duration-300 ease-out ${
            mobileDrawerOpen ? 'translate-y-0' : 'translate-y-full'
          }`}
          style={{
            maxHeight: '80vh',
            backgroundColor: 'var(--bg-surface)',
            borderTop: '1px solid var(--border-subtle)',
            borderRadius: '16px 16px 0 0',
          }}
        >
          {/* Drag handle */}
          <div className="flex justify-center pt-3 pb-2">
            <div
              className="w-10 h-1 rounded-full"
              style={{ backgroundColor: 'var(--text-muted)' }}
            />
          </div>
          <div className="overflow-y-auto" style={{ maxHeight: 'calc(80vh - 40px)' }}>
            <LeftPanel
              activeSession={activeSession}
              history={history}
              isRunning={isRunning}
              onSubmit={(q) => { startResearch(q); setMobileDrawerOpen(false) }}
              onLoad={(id) => { loadSession(id); setMobileDrawerOpen(false) }}
            />
          </div>
        </div>

        {/* ── Mobile: bottom bar with query + history trigger ── */}
        <div
          className="fixed bottom-0 left-0 right-0 lg:hidden flex items-center gap-2 px-4 py-3 z-30"
          style={{
            backgroundColor: 'var(--bg-surface)',
            borderTop: '1px solid var(--border-subtle)',
          }}
        >
          <button
            onClick={() => setMobileDrawerOpen(true)}
            className="flex items-center gap-2 flex-1 text-sm px-4 py-2.5 rounded-lg text-left"
            style={{
              backgroundColor: 'var(--bg-raised)',
              color: 'var(--text-muted)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <svg
              viewBox="0 0 20 20"
              fill="none"
              stroke="currentColor"
              strokeWidth={1.5}
              className="w-4 h-4 shrink-0"
              style={{ color: 'var(--text-muted)' }}
            >
              <circle cx="9" cy="9" r="6" />
              <path strokeLinecap="round" d="M15 15l3 3" />
            </svg>
            What do you want to research?
          </button>
        </div>
      </div>
    </div>
  )
}
