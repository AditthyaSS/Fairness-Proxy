import { Handle, Position } from '@xyflow/react';
import { usePipelineStore } from '../../store/usePipelineStore';

export function ClientRequestNode() {
  const active = usePipelineStore((s) => s.activeStages.has(0));
  const result = usePipelineStore((s) => s.result);
  const error = usePipelineStore((s) => s.error);
  const scenario = usePipelineStore((s) => s.scenario);
  const formValues = usePipelineStore((s) => s.formValues);

  const hasError = error !== null;
  const borderColor = hasError ? '#ff4757' : active ? '#6b7280' : '#242830';
  const keyCount = Object.keys(formValues).length;

  return (
    <div
      className={`node-card ${active ? 'node-active' : ''}`}
      style={{
        borderColor,
        width: 280,
        boxShadow: hasError ? '0 0 20px rgba(255,71,87,0.3)' : active ? '0 0 15px rgba(107,114,128,0.2)' : 'none',
      }}
    >
      <div className="node-header" style={{ color: '#6b7280' }}>
        <span className="node-icon">📡</span>
        CLIENT REQUEST
      </div>
      <div className="node-body">
        <span className="badge badge-gray">{scenario.replace('_', ' ')}</span>
        <span className="badge badge-gray">{keyCount} fields</span>
        {hasError && (
          <div style={{ color: '#ff4757', fontSize: 11, marginTop: 6 }}>
            Cannot connect — is the server running?
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} className="flow-handle" />
    </div>
  );
}
