import { useState } from 'react'
import FindingsStream from './RightPanel/FindingsStream'
import CriticCard from './RightPanel/CriticCard'
import EvaluationCard from './RightPanel/EvaluationCard'

const TABS = ['Findings', 'Analysis']

/**
 * RightPanel — 320px fixed width, tabbed: Findings | Analysis.
 * Props:
 *   session — WorkflowState | null
 */
export default function RightPanel({ session }) {
  const [activeTab, setActiveTab] = useState('Findings')

  const results     = session?.results ?? []
  const criticResult = session?.critic_result ?? null
  const evaluations  = session?.evaluations ?? []

  const hasAnalysis = criticResult || evaluations.length > 0

  return (
    <aside
      className="flex flex-col shrink-0 h-full"
      style={{
        width: '320px',
        borderLeft: '1px solid var(--border-subtle)',
        backgroundColor: 'var(--bg-surface)',
      }}
    >
      {/* Tab bar */}
      <div
        className="flex shrink-0"
        style={{ borderBottom: '1px solid var(--border-subtle)' }}
      >
        {TABS.map((tab) => {
          const isActive = activeTab === tab
          return (
            <button
              key={tab}
              id={`tab-${tab.toLowerCase()}`}
              onClick={() => setActiveTab(tab)}
              className="relative flex items-center gap-1.5 px-5 py-3 text-xs font-medium transition-colors duration-150 focus-ring"
              style={{
                color: isActive ? 'var(--text-primary)' : 'var(--text-muted)',
                backgroundColor: 'transparent',
                borderBottom: isActive ? '2px solid var(--accent)' : '2px solid transparent',
              }}
            >
              {tab}
              {tab === 'Findings' && results.length > 0 && (
                <span
                  className="text-xs font-mono px-1.5 py-0.5 rounded"
                  style={{
                    backgroundColor: 'var(--accent-dim)',
                    color: 'var(--accent)',
                    fontSize: '10px',
                  }}
                >
                  {results.length}
                </span>
              )}
              {tab === 'Analysis' && hasAnalysis && (
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ backgroundColor: 'var(--accent)' }}
                />
              )}
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      <div className="flex-1 min-h-0 flex flex-col">
        {activeTab === 'Findings' ? (
          <FindingsStream results={results} />
        ) : (
          <div className="panel-scroll flex-1 p-3 flex flex-col gap-3">
            {!hasAnalysis ? (
              <div className="flex flex-col items-center justify-center h-full">
                <p className="text-xs text-center" style={{ color: 'var(--text-muted)' }}>
                  Analysis data will appear<br />after the critique stage
                </p>
              </div>
            ) : (
              <>
                {criticResult && <CriticCard criticResult={criticResult} />}
                {evaluations.map((ev) => (
                  <EvaluationCard key={ev.iteration} evaluation={ev} />
                ))}
              </>
            )}
          </div>
        )}
      </div>
    </aside>
  )
}
