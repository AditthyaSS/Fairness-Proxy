import { Handle, Position } from '@xyflow/react';
import { motion } from 'framer-motion';
import { usePipelineStore } from '../../store/usePipelineStore';

const SLOTS = ['Orig', 'T1', 'T2', 'T3', 'T4', 'T5'];

export function UpstreamNode() {
  const active = usePipelineStore((s) => s.activeStages.has(3));
  const completed = usePipelineStore((s) => s.completedStages.has(3));
  const stageScores = usePipelineStore((s) => s.stageData.upstream_scores);
  const result = usePipelineStore((s) => s.result);

  const wsScores = stageScores ?? [];
  const inferences = result?.upstream_inferences ?? [];
  const verdict = result?.rl_decision?.verdict;
  const hasBias = completed && verdict && verdict !== 'PASS';

  // Get original score to compute deltas
  const origWs = wsScores.find((s) => s.index === 0);
  const origScore = origWs?.score ?? inferences[0]?.decision_score ?? null;

  const color = '#64748b';
  const borderColor = hasBias ? (verdict === 'BLOCK' ? '#ff4757' : '#f5a623') : active ? color : '#242830';
  const glowColor = hasBias ? (verdict === 'BLOCK' ? '#ff475740' : '#f5a62340') : `${color}33`;

  return (
    <div
      className={`node-card ${active && !completed ? 'node-pulse' : ''}`}
      style={{
        borderColor,
        width: 480,
        boxShadow: active || hasBias ? `0 0 20px ${glowColor}` : 'none',
      }}
    >
      <Handle type="target" position={Position.Top} className="flow-handle" />
      <div className="node-header" style={{ color: hasBias ? borderColor : color }}>
        <span className="node-icon">⚡</span>
        UPSTREAM AI × {SLOTS.length}
        {hasBias && (
          <span className="bias-indicator">
            ⚠ SCORE DISPARITY DETECTED
          </span>
        )}
      </div>
      <div className="node-body">
        <div className="upstream-grid">
          {SLOTS.map((label, i) => {
            const wsEntry = wsScores.find((s) => s.index === i);
            const finalEntry = inferences[i];
            const score = wsEntry?.score ?? (completed ? finalEntry?.decision_score : null);
            const hasScore = score !== null && score !== undefined;

            // Compute delta from original
            const delta = hasScore && origScore !== null ? Math.abs(score - origScore) : 0;
            const isHighDelta = i > 0 && delta > 0.15; // Significant divergence
            const barColor = i === 0
              ? '#e2e8f0'
              : isHighDelta
                ? (delta > 0.3 ? '#ff4757' : '#f5a623')
                : '#64748b';

            return (
              <div key={i} className="upstream-bar-container">
                <div className="upstream-label">{label}</div>
                <div
                  className="upstream-bar-bg"
                  style={{
                    border: isHighDelta ? `1px solid ${barColor}44` : undefined,
                  }}
                >
                  {hasScore ? (
                    <motion.div
                      className="upstream-bar-fill"
                      initial={{ height: 0 }}
                      animate={{ height: `${score * 100}%` }}
                      transition={{ duration: 0.5, ease: 'easeOut' }}
                      style={{
                        background: barColor,
                        position: 'absolute',
                        bottom: 0,
                        width: '100%',
                        borderRadius: '3px 3px 0 0',
                        boxShadow: isHighDelta ? `0 0 6px ${barColor}66` : 'none',
                      }}
                    />
                  ) : (
                    active && (
                      <div
                        className="upstream-bar-fill"
                        style={{
                          height: '20%',
                          background: '#64748b44',
                          position: 'absolute',
                          bottom: 0,
                          width: '100%',
                          animation: 'barPulse 1.2s ease-in-out infinite',
                        }}
                      />
                    )
                  )}
                </div>
                <div
                  className="upstream-score"
                  style={{ color: isHighDelta ? barColor : undefined }}
                >
                  {hasScore ? score.toFixed(3) : active ? '...' : '—'}
                </div>
                {/* Delta indicator */}
                {hasScore && i > 0 && delta > 0.05 && (
                  <div
                    className="upstream-delta"
                    style={{ color: isHighDelta ? barColor : '#6b7280' }}
                  >
                    Δ{delta > 0 && score < origScore! ? '−' : '+'}{delta.toFixed(2)}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        {/* Max delta summary */}
        {completed && result?.fairness_metrics && (
          <div className="upstream-summary">
            <span>max Δ = {result.fairness_metrics.max_score_delta.toFixed(4)}</span>
            <span>L_CF = {result.fairness_metrics.counterfactual_variance.toFixed(4)}</span>
            {hasBias && (
              <span className="bias-flag" style={{ color: borderColor }}>
                ← bias source
              </span>
            )}
          </div>
        )}
        <div className="node-sub">asyncio.gather · parallel inference</div>
      </div>
      <Handle type="source" position={Position.Bottom} className="flow-handle" />
    </div>
  );
}
