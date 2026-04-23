import { Handle, Position } from '@xyflow/react';
import { motion } from 'framer-motion';
import { usePipelineStore } from '../../store/usePipelineStore';
import type { Verdict } from '../../types/api';

const VERDICT_COLORS: Record<Verdict, string> = {
  PASS: '#00d084',
  MITIGATE: '#f5a623',
  BLOCK: '#ff4757',
};

const PHASE_COLORS: Record<string, string> = {
  heuristic_fallback: '#4a5568',
  ppo_warming:        '#7c3aed',
  trained_ppo:        '#00d084',
};

const STATE_LABELS = ['ŷ', 'L_CF', 'Δmax', 'Δmean', 'Σ|wₚ|', 'max|wₚ|', '|1−DPR|'];
const ACTIONS: Verdict[] = ['PASS', 'MITIGATE', 'BLOCK'];

export function RLAgentNode() {
  const active = usePipelineStore((s) => s.activeStages.has(5));
  const completed = usePipelineStore((s) => s.completedStages.has(5));
  const result = usePipelineStore((s) => s.result);
  const rlStats = usePipelineStore((s) => s.rlStats);

  const rl = result?.rl_decision;
  const verdict = rl?.verdict;
  const fm = result?.fairness_metrics;

  // Build 7-dim state vector from result
  const stateVec = completed && fm && result
    ? [
        result.original_score,
        Math.min(fm.counterfactual_variance, 1),
        Math.min(fm.max_score_delta, 1),
        Math.min(fm.mean_score_delta, 1),
        Math.min(fm.xai_report.total_protected_shap, 1),
        Math.min(
          fm.xai_report.feature_importances
            .filter((f) => f.is_protected)
            .reduce((m, f) => Math.max(m, f.abs_shap_value), 0),
          1
        ),
        Math.abs(1 - Math.min(fm.demographic_parity_ratio ?? 1, 2)),
      ]
    : new Array(7).fill(0);

  const maxVal = Math.max(...stateVec, 0.001);

  const color = '#6366f1';
  const borderColor = active ? color : '#242830';

  return (
    <div
      className={`node-card ${active && !completed ? 'node-pulse' : ''}`}
      style={{
        borderColor,
        width: 340,
        boxShadow: active ? `0 0 20px ${color}33` : 'none',
      }}
    >
      <Handle type="target" position={Position.Top} className="flow-handle" />
      <div className="node-header" style={{ color }}>
        <span className="node-icon">🧠</span>
        PPO RL AGENT
        {rlStats && (
          <span style={{ marginLeft: 'auto', fontSize: 9, color: '#6b7280' }}>
            {rlStats.total_episodes} episodes
          </span>
        )}
      </div>
      <div className="node-body">
        {/* State vector mini chart */}
        <div className="state-vector">
          {STATE_LABELS.map((label, i) => (
            <div key={label} className="state-bar-row">
              <span className="state-label">{label}</span>
              <div className="state-bar-bg">
                <motion.div
                  className="state-bar-fill"
                  initial={{ width: 0 }}
                  animate={{
                    width: completed
                      ? `${Math.min((stateVec[i] / maxVal) * 100, 100)}%`
                      : '0%',
                  }}
                  transition={{ duration: 0.4, delay: i * 0.05 }}
                  style={{ background: color, height: '100%', borderRadius: 3 }}
                />
              </div>
              <span className="state-val">
                {completed ? stateVec[i].toFixed(3) : '—'}
              </span>
            </div>
          ))}
        </div>

        {/* Action selector */}
        <div className="action-row">
          {ACTIONS.map((a) => (
            <motion.div
              key={a}
              className="action-pill"
              animate={{
                borderColor: verdict === a ? VERDICT_COLORS[a] : '#242830',
                color: verdict === a ? VERDICT_COLORS[a] : '#6b7280',
                backgroundColor: verdict === a ? `${VERDICT_COLORS[a]}15` : 'transparent',
                scale: verdict === a ? 1.05 : 1,
              }}
              transition={{ duration: 0.3 }}
            >
              {a}
            </motion.div>
          ))}
        </div>

        {/* Reward curve sparkline */}
        {rlStats?.reward_curve && rlStats.reward_curve.length > 1 && (
          <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 9, color: '#6b7280', fontFamily: 'var(--font-mono)' }}>
              R̄={rlStats.avg_reward_last50.toFixed(3)}
            </span>
            <svg width="80" height="16" viewBox={`0 0 ${rlStats.reward_curve.length} 1`} preserveAspectRatio="none" style={{ flex: 1 }}>
              <polyline
                points={rlStats.reward_curve.map((v, i) => {
                  const min = Math.min(...rlStats.reward_curve);
                  const max = Math.max(...rlStats.reward_curve);
                  const norm = max === min ? 0.5 : (v - min) / (max - min);
                  return `${i},${1 - norm}`;
                }).join(' ')}
                fill="none"
                stroke="#7c3aed"
                strokeWidth="0.04"
                vectorEffect="non-scaling-stroke"
              />
            </svg>
          </div>
        )}

        {/* Agent phase indicator */}
        <div className="rl-node-phase">
          <div
            className="rl-node-phase-dot"
            style={{
              background: PHASE_COLORS[rlStats?.agent_phase ?? 'heuristic_fallback']
                ?? '#4a5568',
            }}
          />
          <span style={{
            color: PHASE_COLORS[rlStats?.agent_phase ?? 'heuristic_fallback']
              ?? '#4a5568',
            fontSize: 9,
            fontFamily: 'var(--font-mono)',
          }}>
            {rlStats?.agent_phase === 'trained_ppo' ? 'Trained PPO'
              : rlStats?.agent_phase === 'ppo_warming' ? 'PPO Warming'
              : 'Heuristic'}
          </span>
          {rlStats && rlStats.agent_phase !== 'trained_ppo' && (
            <span style={{ fontSize: 8, color: '#4b5563', fontFamily: 'var(--font-mono)' }}>
              {rlStats.total_labeled_episodes ?? 0} eps
            </span>
          )}
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className="flow-handle" />
    </div>
  );
}
