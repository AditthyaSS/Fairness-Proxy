import { usePipelineStore } from '../store/usePipelineStore';
import { usePipelineRun } from '../hooks/usePipelineRun';

export function AutoStreamControls() {
  const autoStream = usePipelineStore((s) => s.autoStream);
  const setAutoStream = usePipelineStore((s) => s.setAutoStream);
  const runState = usePipelineStore((s) => s.runState);
  const { stop } = usePipelineRun();

  return (
    <div className="auto-stream-panel">
      <div className="auto-stream-header">
        <span>⚡ Auto-stream dataset</span>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={autoStream}
            onChange={(e) => {
              if (!e.target.checked) stop();
              setAutoStream(e.target.checked);
            }}
          />
          <span className="toggle-slider" />
        </label>
      </div>
      {autoStream && (
        <div className="auto-stream-status">
          <span className={`status-dot ${runState === 'running' ? 'status-dot-active' : ''}`} />
          <span style={{ fontSize: 10, color: '#9ca3af' }}>
            {runState === 'running' ? 'Processing...' : 'Waiting for next row...'}
          </span>
        </div>
      )}
    </div>
  );
}
