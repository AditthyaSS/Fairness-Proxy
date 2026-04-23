import { Handle, Position } from '@xyflow/react';
import { motion } from 'framer-motion';
import { usePipelineStore } from '../../store/usePipelineStore';

export function FairnessEvaluatorNode() {
  const active = usePipelineStore((s) => s.activeStages.has(4));
  const completed = usePipelineStore((s) => s.completedStages.has(4));
  const result = usePipelineStore((s) => s.result);

  const fm = result?.fairness_metrics;
  const lcf = fm?.counterfactual_variance;
  const shapSum = fm?.xai_report?.total_protected_shap;
  const reward = fm?.rl_reward;
  const dpr = fm?.demographic_parity_ratio;
  const verdict = result?.rl_decision?.verdict;

  // Bias severity calculation
  const hasBias = completed && verdict && verdict !== 'PASS';
  const biasLevel = verdict === 'BLOCK' ? 'severe' : verdict === 'MITIGATE' ? 'moderate' : 'none';

  // Top biased features
  const biasedFeatures = fm?.xai_report?.feature_importances
    ?.filter((f) => f.is_protected && f.abs_shap_value > 0.01)
    ?.sort((a, b) => b.abs_shap_value - a.abs_shap_value)
    ?.slice(0, 3) ?? [];

  // Severity gauge (0-100)
  const severityPct = completed ? Math.min(
    ((lcf ?? 0) * 400 + (shapSum ?? 0) * 300),
    100
  ) : 0;
  const severityColor = severityPct > 60 ? '#ff4757' : severityPct > 25 ? '#f5a623' : '#00d084';

  const color = hasBias ? (biasLevel === 'severe' ? '#ff4757' : '#f5a623') : '#f5a623';
  const borderColor = active || hasBias ? color : '#242830';

  return (
    <div
      className={`node-card ${active && !completed ? 'node-pulse' : ''}`}
      style={{
        borderColor,
        width: 400,
        boxShadow: active || hasBias ? `0 0 20px ${color}33` : 'none',
      }}
    >
      <Handle type="target" position={Position.Top} className="flow-handle" />
      <div className="node-header" style={{ color }}>
        <span className="node-icon">⚖️</span>
        FAIRNESS EVALUATOR
        {hasBias && (
          <span className="bias-indicator">
            {biasLevel === 'severe' ? '🛑 SEVERE BIAS' : '⚠ BIAS DETECTED'}
          </span>
        )}
      </div>
      <div className="node-body">
        {/* Bias severity gauge */}
        {completed && (
          <div className="severity-gauge">
            <div className="severity-track">
              <motion.div
                className="severity-fill"
                initial={{ width: 0 }}
                animate={{ width: `${severityPct}%` }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
                style={{ background: `linear-gradient(90deg, #00d084, ${severityColor})` }}
              />
              <div className="severity-markers">
                <span style={{ left: '25%' }} />
                <span style={{ left: '60%' }} />
              </div>
            </div>
            <div className="severity-labels">
              <span style={{ color: '#00d084' }}>Fair</span>
              <span style={{ color: '#f5a623' }}>Biased</span>
              <span style={{ color: '#ff4757' }}>Severe</span>
            </div>
          </div>
        )}

        <div className="evaluator-panels">
          <div className={`eval-panel ${completed && lcf !== undefined && lcf > 0.05 ? 'eval-panel-warn' : ''}`}>
            <div className="eval-title" title="(1/N)·Σ(ŷ−ŷ'ᵢ)²">
              L_CF <span className="tooltip-icon">ℹ</span>
            </div>
            <div className={`eval-value ${completed && lcf !== undefined && lcf > 0.05 ? 'val-warn' : ''}`}>
              {completed && lcf !== undefined ? lcf.toFixed(6) : '—'}
            </div>
          </div>
          <div className={`eval-panel ${completed && shapSum !== undefined && shapSum > 0.15 ? 'eval-panel-danger' : ''}`}>
            <div className="eval-title">
              Σ|wₚ| SHAP
            </div>
            <div className={`eval-value ${completed && shapSum !== undefined && shapSum > 0.15 ? 'val-danger' : ''}`}>
              {completed && shapSum !== undefined ? shapSum.toFixed(4) : '—'}
            </div>
          </div>
        </div>

        {/* Top biased features callout */}
        {biasedFeatures.length > 0 && (
          <div className="bias-features">
            <div className="bias-features-label">⚠ Biased features detected:</div>
            <div className="bias-features-list">
              {biasedFeatures.map((f) => (
                <div key={f.feature_name} className="bias-feature-item">
                  <span className="bias-feature-name">{f.feature_name}</span>
                  <div className="bias-feature-bar-bg">
                    <motion.div
                      className="bias-feature-bar"
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(f.abs_shap_value * 500, 100)}%` }}
                      transition={{ duration: 0.5 }}
                      style={{
                        background: f.abs_shap_value > 0.1 ? '#ff4757' : '#f5a623',
                      }}
                    />
                  </div>
                  <span className="bias-feature-val">{f.abs_shap_value.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="node-row" style={{ marginTop: 8 }}>
          <span className="node-label">RL Reward R</span>
          <span className="node-value mono">{completed && reward !== undefined ? reward.toFixed(4) : '—'}</span>
        </div>
        <div className="node-row">
          <span className="node-label">DPR Ratio</span>
          <span className="node-value mono">{completed && dpr !== undefined && dpr !== null ? dpr.toFixed(3) : '—'}</span>
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className="flow-handle" />
    </div>
  );
}
