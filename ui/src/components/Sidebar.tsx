import { useEffect } from 'react';
import { usePipelineStore } from '../store/usePipelineStore';
import { usePipelineRun } from '../hooks/usePipelineRun';
import { AutoStreamControls } from './AutoStreamControls';
import { SCENARIOS, SCENARIO_ORDER } from '../constants/scenarios';

export function Sidebar() {
  const scenario = usePipelineStore((s) => s.scenario);
  const setScenario = usePipelineStore((s) => s.setScenario);
  const formValues = usePipelineStore((s) => s.formValues);
  const setFormValues = usePipelineStore((s) => s.setFormValues);
  const updateField = usePipelineStore((s) => s.updateField);
  const mockEnabled = usePipelineStore((s) => s.mockEnabled);
  const setMockEnabled = usePipelineStore((s) => s.setMockEnabled);
  const baseUrl = usePipelineStore((s) => s.baseUrl);
  const setBaseUrl = usePipelineStore((s) => s.setBaseUrl);
  const runState = usePipelineStore((s) => s.runState);
  const error = usePipelineStore((s) => s.error);
  const { run } = usePipelineRun();

  const preset = SCENARIOS[scenario];

  // Initialize form values on scenario change
  useEffect(() => {
    if (preset) {
      const values: Record<string, any> = {};
      for (const [key, def] of Object.entries(preset.fields)) {
        values[key] = def.value;
      }
      setFormValues(values);
    }
  }, [scenario, setFormValues]);

  const isRunning = runState === 'running';

  return (
    <div className="sidebar">
      {/* Scenario Tabs */}
      <div className="sidebar-section">
        <div className="sidebar-label">SCENARIO</div>
        <div className="scenario-tabs">
          {SCENARIO_ORDER.map((s) => (
            <button
              key={s}
              className={`scenario-pill ${scenario === s ? 'scenario-active' : ''}`}
              onClick={() => setScenario(s)}
              disabled={isRunning}
            >
              {SCENARIOS[s].label}
              {SCENARIOS[s].liveDataset && (
                <span className="live-badge">LIVE</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Form Fields */}
      <div className="sidebar-section form-section">
        <div className="sidebar-label">INPUT PAYLOAD</div>
        {preset?.liveDataset && (
          <div className="dataset-hint">
            Real UCI Adult rows auto-fetched when you run
          </div>
        )}
        <div className="form-fields">
          {Object.entries(formValues).map(([key, value]) => {
            const fieldDef = preset?.fields[key];
            const isMerit = fieldDef?.type === 'merit';
            const isProtected = fieldDef?.type === 'protected';
            return (
              <div
                key={key}
                className={`form-field ${isMerit ? 'field-merit' : ''} ${isProtected ? 'field-protected' : ''}`}
              >
                <label className="field-label">
                  {key}
                  {isProtected && <span className="field-badge-protected">P</span>}
                  {isMerit && <span className="field-badge-merit">M</span>}
                </label>
                <input
                  className="field-input"
                  type={typeof value === 'number' ? 'number' : 'text'}
                  value={typeof value === 'boolean' ? String(value) : value}
                  onChange={(e) => {
                    const v =
                      typeof value === 'number'
                        ? Number(e.target.value)
                        : typeof value === 'boolean'
                        ? e.target.value === 'true'
                        : e.target.value;
                    updateField(key, v);
                  }}
                  disabled={isRunning}
                />
              </div>
            );
          })}
        </div>
      </div>

      {/* Bias Toggle */}
      <div className="sidebar-section">
        <div className="bias-toggle-card">
          <div className="bias-toggle-header">
            <span>⚠️ Simulate biased upstream</span>
          </div>
          <div className="bias-toggle-desc">
            Injects a discriminatory mock AI that scores by race/gender
          </div>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={mockEnabled}
              onChange={(e) => setMockEnabled(e.target.checked)}
              disabled={isRunning}
            />
            <span className="toggle-slider" />
            <span className="toggle-label">{mockEnabled ? 'ON' : 'OFF'}</span>
          </label>
        </div>
      </div>

      {/* Auto Stream */}
      {preset?.liveDataset && (
        <div className="sidebar-section">
          <AutoStreamControls />
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="error-banner">
          <span>❌ {error}</span>
          <button onClick={() => usePipelineStore.getState().setError(null)}>×</button>
        </div>
      )}

      {/* Run Button */}
      <div className="sidebar-section">
        <button
          className={`run-button ${isRunning ? 'run-button-running' : ''}`}
          onClick={() => run()}
          disabled={isRunning}
        >
          {isRunning ? (
            <span className="run-pulse">⠿ PROCESSING...</span>
          ) : (
            <span>▶ RUN PIPELINE</span>
          )}
        </button>
      </div>

      {/* API Base URL */}
      <div className="sidebar-section sidebar-footer">
        <div className="sidebar-label">API BASE URL</div>
        <input
          className="field-input"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          disabled={isRunning}
        />
      </div>
    </div>
  );
}
