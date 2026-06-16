import { stageColor } from '../../utils/formatters'

// The ordered pipeline stages and their display labels
const PIPELINE_STAGES = [
  { key: 'planning',    label: 'Planning' },
  { key: 'researching', label: 'Researching' },
  { key: 'critiquing',  label: 'Critiquing' },
  { key: 'reporting',   label: 'Reporting' },
  { key: 'evaluating',  label: 'Evaluating' },
]

// Which stages come before a given stage in the pipeline
const STAGE_ORDER = PIPELINE_STAGES.map((s) => s.key)

function getStageStatus(stageKey, currentStage, overallStatus) {
  const currentIdx = STAGE_ORDER.indexOf(currentStage)
  const stageIdx   = STAGE_ORDER.indexOf(stageKey)

  if (overallStatus === 'completed') return 'complete'
  if (overallStatus === 'failed' && currentStage === stageKey) return 'failed'
  if (stageIdx < currentIdx) return 'complete'
  if (stageIdx === currentIdx) return 'active'
  return 'pending'
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth={2.5} className="w-3 h-3">
      <polyline points="2,6 5,9 10,3" />
    </svg>
  )
}

function StageNode({ stage, status, isLast }) {
  const color = stageColor(stage.key)

  const nodeStyle = (() => {
    switch (status) {
      case 'active':
        return {
          backgroundColor: color,
          border: `2px solid ${color}`,
          color: '#fff',
          animation: 'pulse-ring 1.5s ease-out infinite',
          boxShadow: `0 0 0 0 ${color}`,
          animationName: 'pulse-ring',
        }
      case 'complete':
        return {
          backgroundColor: color,
          border: `2px solid ${color}`,
          color: '#fff',
        }
      case 'failed':
        return {
          backgroundColor: 'var(--stage-failed)',
          border: '2px solid var(--stage-failed)',
          color: '#fff',
        }
      default: // pending
        return {
          backgroundColor: 'transparent',
          border: '2px solid var(--text-muted)',
          color: 'var(--text-muted)',
        }
    }
  })()

  return (
    <div className="flex flex-col items-center gap-1.5 relative z-10">
      {/* Circle node */}
      <div
        className="w-6 h-6 rounded-full flex items-center justify-center shrink-0 animate-pulse-ring"
        style={{
          ...nodeStyle,
          animation: status === 'active' ? 'pulse-ring 1.5s ease-out infinite' : 'none',
        }}
      >
        {status === 'complete' && <CheckIcon />}
        {status === 'active' && (
          <span
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: '#fff' }}
          />
        )}
        {status === 'failed' && (
          <span className="text-xs font-bold" style={{ lineHeight: 1 }}>✕</span>
        )}
      </div>

      {/* Label */}
      <span
        className="text-xs whitespace-nowrap"
        style={{
          color: status === 'pending' ? 'var(--text-muted)' : 'var(--text-secondary)',
          fontSize: '10px',
        }}
      >
        {stage.label}
      </span>
    </div>
  )
}

function StageConnector({ fromStatus, toStatus, fromStage }) {
  const isFilled = fromStatus === 'complete' || toStatus === 'complete' || toStatus === 'active'
  const color = stageColor(fromStage)

  return (
    <div
      className="flex-1 relative mt-[-10px]"  // align to node center
      style={{ height: '2px', marginBottom: '22px' }}
    >
      {/* Background track */}
      <div
        className="absolute inset-0 rounded-full"
        style={{ backgroundColor: 'var(--bg-hover)' }}
      />
      {/* Fill overlay */}
      <div
        className="absolute inset-0 rounded-full stage-line"
        style={{
          backgroundColor: isFilled ? color : 'transparent',
          width: isFilled ? '100%' : '0%',
        }}
      />
    </div>
  )
}

/**
 * PipelineTrack — horizontal row of 5 stage nodes with connecting lines.
 * Props:
 *   currentStage  — string
 *   status        — "running" | "completed" | "failed"
 */
export default function PipelineTrack({ currentStage, status }) {
  return (
    <div className="flex items-end gap-0 px-2">
      {PIPELINE_STAGES.map((stage, idx) => {
        const stageStatus = getStageStatus(stage.key, currentStage, status)
        const isLast = idx === PIPELINE_STAGES.length - 1

        return (
          <div key={stage.key} className="flex items-end flex-1">
            <StageNode stage={stage} status={stageStatus} isLast={isLast} />
            {!isLast && (
              <StageConnector
                fromStage={stage.key}
                fromStatus={stageStatus}
                toStatus={getStageStatus(PIPELINE_STAGES[idx + 1].key, currentStage, status)}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}
