import { Handle, Position } from '@xyflow/react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePipelineStore } from '../../store/usePipelineStore';
import type { Verdict } from '../../types/api';

const VERDICT_CONFIG: Record<Verdict, { color: string; label: string }> = {
  PASS: { color: '#00d084', label: 'Decision forwarded unchanged' },
  MITIGATE: { color: '#f5a623', label: 'Score corrected to reduce bias' },
  BLOCK: { color: '#ff4757', label: 'Request suppressed · bias too severe' },
};

export function VerdictNode() {
  const active = usePipelineStore((s) => s.activeStages.has(6));
  const completed = usePipelineStore((s) => s.completedStages.has(6));
  const result = usePipelineStore((s) => s.result);

  const rl = result?.rl_decision;
  const verdict = rl?.verdict;
  const config = verdict ? VERDICT_CONFIG[verdict] : null;

  return (
    <div
      className="node-card"
      style={{
        borderColor: config ? config.color : '#242830',
        width: 440,
        boxShadow: config ? `0 0 30px ${config.color}40` : 'none',
        transition: 'all 0.5s ease-out',
      }}
    >
      <Handle type="target" position={Position.Top} className="flow-handle" />
      <AnimatePresence>
        {completed && verdict && config ? (
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: 'spring', duration: 0.6 }}
          >
            <div className="node-header" style={{ color: config.color, fontSize: 16 }}>
              <span className="node-icon">
                {verdict === 'PASS' ? '✅' : verdict === 'MITIGATE' ? '⚠️' : '🛑'}
              </span>
              VERDICT: {verdict}
            </div>
            <div className="node-body">
              <div style={{ color: '#9ca3af', fontSize: 12, marginBottom: 8 }}>
                {config.label}
              </div>
              {verdict === 'MITIGATE' && result && (
                <div className="score-correction">
                  <span className="score-old">{result.original_score.toFixed(4)}</span>
                  <span className="score-arrow">→</span>
                  <span className="score-new" style={{ color: config.color }}>
                    {result.final_score.toFixed(4)}
                  </span>
                </div>
              )}
              {rl?.explanation && (
                <div className="node-sub" style={{ marginTop: 6, color: '#6b7280' }}>
                  {rl.explanation}
                </div>
              )}
            </div>
          </motion.div>
        ) : (
          <div>
            <div className="node-header" style={{ color: '#6b7280' }}>
              <span className="node-icon">⏳</span>
              VERDICT
            </div>
            <div className="node-body">
              <span style={{ color: '#4b5563', fontSize: 12 }}>Awaiting pipeline result...</span>
            </div>
          </div>
        )}
      </AnimatePresence>
      <Handle type="source" position={Position.Bottom} className="flow-handle" />
    </div>
  );
}
