import { ReactFlowProvider } from '@xyflow/react';
import { Sidebar } from './components/Sidebar';
import { FlowCanvas } from './components/FlowCanvas';
import { BottomDrawer } from './components/drawer/BottomDrawer';
import { useAuditStats } from './hooks/useAuditStats';
import { useRLStats } from './hooks/useRLStats';
import { useTunnelInfo } from './hooks/useTunnelInfo';
import { usePipelineStore } from './store/usePipelineStore';
import './index.css';

function Header() {
  const apiOnline = usePipelineStore((s) => s.apiOnline);
  const auditStats = usePipelineStore((s) => s.auditStats);
  const rlStats = usePipelineStore((s) => s.rlStats);

  return (
    <header className="app-header">
      <div className="header-left">
        <span className="header-logo">⚖️</span>
        <span className="header-title">Fairness Proxy</span>
        <span className="header-subtitle">Pipeline Visualizer</span>
        <span className={`status-pill ${apiOnline ? 'status-online' : 'status-offline'}`}>
          <span className="status-dot" />
          {apiOnline ? 'API Online' : 'Offline'}
        </span>
        {rlStats && (
          <span className="status-pill" style={{ background: 'rgba(124,58,237,0.12)', color: '#7c3aed' }}>
            <span style={{ fontSize: 9 }}>🧠</span>
            {rlStats.total_episodes} episodes
          </span>
        )}
      </div>
      <div className="header-right">
        {auditStats && (
          <>
            <div className="stat-chip stat-pass">
              PASS <span className="stat-count">{auditStats.pass_count}</span>
            </div>
            <div className="stat-chip stat-mitigate">
              MITIGATE <span className="stat-count">{auditStats.mitigate_count}</span>
            </div>
            <div className="stat-chip stat-block">
              BLOCK <span className="stat-count">{auditStats.block_count}</span>
            </div>
            <div className="stat-chip stat-latency">
              avg <span className="stat-count">{auditStats.avg_latency_ms?.toFixed(0) || 0}ms</span>
            </div>
          </>
        )}
      </div>
    </header>
  );
}

function App() {
  useAuditStats();
  useRLStats();
  useTunnelInfo();

  return (
    <ReactFlowProvider>
      <div className="app-container">
        <Header />
        <div className="app-body">
          <Sidebar />
          <div className="main-area">
            <div className="canvas-wrapper">
              <FlowCanvas />
            </div>
            <BottomDrawer />
          </div>
        </div>
      </div>
    </ReactFlowProvider>
  );
}

export default App;
