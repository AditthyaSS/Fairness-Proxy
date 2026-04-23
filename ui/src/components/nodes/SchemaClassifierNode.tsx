import { Handle, Position } from '@xyflow/react';
import { usePipelineStore } from '../../store/usePipelineStore';

export function SchemaClassifierNode() {
  const active = usePipelineStore((s) => s.activeStages.has(1));
  const completed = usePipelineStore((s) => s.completedStages.has(1));
  const result = usePipelineStore((s) => s.result);
  const stageData = usePipelineStore((s) => s.stageData.classified);

  const schema = result?.classified_schema;
  const domain = stageData?.domain ?? schema?.domain ?? '—';
  const confidence = stageData?.confidence ?? schema?.domain_confidence;
  const meritKeys = stageData?.merit_features ?? (schema ? Object.keys(schema.merit_features) : []);
  const protectedKeys = stageData?.protected_features ?? (schema ? Object.keys(schema.protected_features) : []);

  // Get which protected features contributed to bias (from SHAP)
  const verdict = result?.rl_decision?.verdict;
  const hasBias = completed && verdict && verdict !== 'PASS';
  const biasedFeatures = result?.fairness_metrics?.xai_report?.feature_importances
    ?.filter((f) => f.is_protected && f.abs_shap_value > 0.01)
    ?.map((f) => f.feature_name) ?? [];

  const color = '#4f8ef7';
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
        <span className="node-icon">🔬</span>
        SCHEMA CLASSIFIER
      </div>
      <div className="node-body">
        <div className="node-row">
          <span className="node-label">Domain</span>
          <span className="badge" style={{ background: `${color}22`, color }}>{domain}</span>
        </div>
        <div className="node-row">
          <span className="node-label">Confidence</span>
          <span className="node-value">
            {confidence !== undefined ? `${(confidence * 100).toFixed(1)}%` : '—'}
          </span>
        </div>
        <div className="node-row" style={{ gap: 6 }}>
          <span className="badge badge-blue">Merit: {meritKeys.length || '—'}</span>
          <span className="badge badge-amber">Protected: {protectedKeys.length || '—'}</span>
        </div>

        {/* Protected feature pills with bias highlighting */}
        {protectedKeys.length > 0 && (
          <div className="schema-features">
            {protectedKeys.map((k: string) => {
              const isBiased = biasedFeatures.includes(k);
              return (
                <span
                  key={k}
                  className={`schema-feature-pill ${isBiased ? 'schema-feature-biased' : ''}`}
                >
                  {isBiased && <span className="bias-dot" />}
                  {k}
                </span>
              );
            })}
          </div>
        )}

        <div className="node-sub">DeBERTa NLI · zero-shot</div>
      </div>
      <Handle type="source" position={Position.Bottom} className="flow-handle" />
    </div>
  );
}
