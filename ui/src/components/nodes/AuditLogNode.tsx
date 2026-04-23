import { Handle, Position } from '@xyflow/react';
import { motion } from 'framer-motion';
import { usePipelineStore } from '../../store/usePipelineStore';
import { useEffect, useState } from 'react';

export function AuditLogNode() {
  const active = usePipelineStore((s) => s.activeStages.has(7));
  const completed = usePipelineStore((s) => s.completedStages.has(7));
  const result = usePipelineStore((s) => s.result);
  const [dimmed, setDimmed] = useState(false);

  useEffect(() => {
    if (completed) {
      const t = setTimeout(() => setDimmed(true), 2000);
      return () => clearTimeout(t);
    }
    setDimmed(false);
  }, [completed]);

  const reqId = result?.request_id || '—';
  const latency = result?.total_latency_ms;

  return (
    <div
      className="node-card"
      style={{
        borderColor: active ? '#6b7280' : '#242830',
        width: 320,
        opacity: dimmed ? 0.5 : 1,
        transition: 'opacity 1s ease-out',
      }}
    >
      <Handle type="target" position={Position.Top} className="flow-handle" />
      {completed ? (
        <motion.div
          initial={{ y: 15, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.4 }}
        >
          <div className="node-header" style={{ color: '#6b7280' }}>
            <span className="node-icon">📋</span>
            AUDIT LOG
          </div>
          <div className="node-body">
            <div className="node-row">
              <span className="node-label">request_id</span>
              <span className="node-value mono" style={{ fontSize: 10 }}>
                {reqId.substring(0, 18)}...
              </span>
            </div>
            <div className="node-row">
              <span className="node-label">latency</span>
              <span className="node-value mono">
                {latency !== undefined ? `${latency.toFixed(0)}ms` : '—'}
              </span>
            </div>
            <div className="node-sub" style={{ color: dimmed ? '#374151' : '#6b7280' }}>
              {dimmed ? '✓ persisted' : 'writing to audit_logs...'}
            </div>
          </div>
        </motion.div>
      ) : (
        <div>
          <div className="node-header" style={{ color: '#374151' }}>
            <span className="node-icon">📋</span>
            AUDIT LOG
          </div>
          <div className="node-body">
            <span style={{ color: '#374151', fontSize: 11 }}>Waiting for verdict...</span>
          </div>
        </div>
      )}
    </div>
  );
}
