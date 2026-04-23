import { Handle, Position } from '@xyflow/react';
import { usePipelineStore } from '../../store/usePipelineStore';

export function TwinGeneratorNode() {
  const active = usePipelineStore((s) => s.activeStages.has(2));
  const completed = usePipelineStore((s) => s.completedStages.has(2));
  const result = usePipelineStore((s) => s.result);
  const stageData = usePipelineStore((s) => s.stageData.twins_generated);
  const classified = usePipelineStore((s) => s.stageData.classified);

  const schema = result?.classified_schema;
  const protectedKeys = classified?.protected_features ?? (schema ? Object.keys(schema.protected_features) : []);
  const meritKeys = classified?.merit_features ?? (schema ? Object.keys(schema.merit_features) : []);
  const twinCount = stageData?.num_twins ?? result?.twin_generation?.twins?.length ?? 5;

  const color = '#7c3aed';
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
        <span className="node-icon">👥</span>
        TWIN GENERATOR
      </div>
      <div className="node-body">
        <div className="node-row">
          <span className="badge" style={{ background: `${color}22`, color }}>
            N={twinCount} twins
          </span>
        </div>
        {protectedKeys.length > 0 && (
          <div className="tag-row">
            {protectedKeys.map((k) => (
              <span key={k} className="tag tag-amber">{k}</span>
            ))}
          </div>
        )}
        {meritKeys.length > 0 && (
          <div className="node-row" style={{ marginTop: 4 }}>
            <span style={{ fontSize: 10, color: '#6b7280' }}>
              🔒 {meritKeys.join(', ')} locked
            </span>
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} className="flow-handle" />
    </div>
  );
}
