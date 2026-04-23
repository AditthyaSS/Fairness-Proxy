import { useCallback, useRef, useEffect } from 'react';
import { usePipelineStore } from '../store/usePipelineStore';
import { SCENARIOS } from '../constants/scenarios';

export function usePipelineRun() {
  const wsRef = useRef<WebSocket | null>(null);

  const run = useCallback(async (overridePayload?: Record<string, any>) => {
    const store = usePipelineStore.getState();
    if (store.runState === 'running') return;

    // Reset
    store.resetPipeline();
    store.setRunState('running');

    const scenario = store.scenario;
    const sc = SCENARIOS[scenario];
    let payload = overridePayload ?? { ...store.formValues };
    let trueLabel: number | null = null;

    // If UCI Adult dataset mode, fetch next real row + true_label
    if (sc?.liveDataset && !overridePayload) {
      try {
        const res = await fetch(
          `${store.baseUrl}/v1/dataset/next?bias_demo=${store.mockEnabled}`
        );
        const row = await res.json();
        if (row.error) {
          store.setError(row.error);
          store.setRunState('error');
          return;
        }
        payload = row.payload;
        trueLabel = row.true_label ?? null;
        store.setFormValues(payload);
      } catch (e) {
        store.setError('Failed to fetch dataset row');
        store.setRunState('error');
        return;
      }
    }

    // Try WebSocket first
    const wsUrl = store.baseUrl.replace('http', 'ws') + '/ws/pipeline';
    let wsConnected = false;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          ws.close();
          reject(new Error('WebSocket connection timeout'));
        }, 5000);

        ws.onopen = () => {
          clearTimeout(timeout);
          wsConnected = true;
          ws.send(JSON.stringify({
            target_endpoint: sc?.targetEndpoint ?? '/v1/decisions/infer',
            domain: scenario,
            payload,
            mock_upstream: store.mockEnabled,
            true_label: trueLabel,
          }));
          store.activateStage(0);
          resolve();
        };

        ws.onerror = () => {
          clearTimeout(timeout);
          reject(new Error('WebSocket failed'));
        };
      });

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        const { stage, data } = msg;
        const s = usePipelineStore.getState();

        switch (stage) {
          case 'classifying':
            s.activateStage(1);
            break;
          case 'classified':
            s.setStageData({ classified: data });
            s.completeStage(1);
            s.activateStage(2);
            break;
          case 'generating_twins':
            s.activateStage(2);
            break;
          case 'twins_generated':
            s.setStageData({ twins_generated: data });
            s.completeStage(2);
            s.activateStage(3);
            break;
          case 'upstream_running':
            s.activateStage(3);
            break;
          case 'upstream_score': {
            const existing = s.stageData.upstream_scores ?? [];
            s.setStageData({ upstream_scores: [...existing, data] });
            break;
          }
          case 'evaluating':
            s.completeStage(3);
            s.activateStage(4);
            setTimeout(() => usePipelineStore.getState().activateStage(5), 400);
            break;
          case 'complete':
            for (let i = 0; i <= 7; i++) {
              s.activateStage(i);
              s.completeStage(i);
            }
            s.setResult(data);
            s.setAccuracyGain(data.accuracy_gain ?? null);
            s.setCumulativeImprovement(data.cumulative_accuracy_improvement ?? null);
            s.setRunState('complete');
            // Open drawer to RL tab to show accuracy
            setTimeout(() => {
              const st = usePipelineStore.getState();
              st.setDrawerOpen(true);
              st.setDrawerTab('rl');
            }, 600);
            ws.close();
            break;
          case 'error':
            s.setError(data.message);
            s.setRunState('error');
            ws.close();
            break;
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
      };

    } catch {
      // WebSocket failed — fall back to HTTP POST
      if (!wsConnected) {
        await runHTTPFallback(payload, scenario, sc, trueLabel);
      }
    }
  }, []);

  // HTTP fallback (same as before)
  const runHTTPFallback = useCallback(async (
    payload: Record<string, any>,
    scenario: string,
    sc: any,
    trueLabel: number | null = null
  ) => {
    const store = usePipelineStore.getState();

    // Staged activation (UX timing)
    const timers: number[] = [];
    timers.push(window.setTimeout(() => usePipelineStore.getState().activateStage(0), 100));
    timers.push(window.setTimeout(() => usePipelineStore.getState().activateStage(1), 300));
    timers.push(window.setTimeout(() => usePipelineStore.getState().activateStage(2), 900));
    timers.push(window.setTimeout(() => usePipelineStore.getState().activateStage(3), 1500));
    timers.push(window.setTimeout(() => usePipelineStore.getState().activateStage(4), 2100));
    timers.push(window.setTimeout(() => usePipelineStore.getState().activateStage(5), 2500));

    try {
      const response = await fetch(
        `${store.baseUrl}/v1/proxy/infer?mock=${store.mockEnabled}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            target_endpoint: sc?.targetEndpoint ?? '/v1/decisions/infer',
            domain: scenario,
            payload,
            true_label: trueLabel,
          }),
        }
      );

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      timers.forEach(clearTimeout);

      const s = usePipelineStore.getState();
      for (let i = 0; i <= 7; i++) {
        s.activateStage(i);
        s.completeStage(i);
      }
      s.setResult(data);
      s.setAccuracyGain(data.accuracy_gain ?? null);
      s.setCumulativeImprovement(data.cumulative_accuracy_improvement ?? null);
      s.setRunState('complete');
      setTimeout(() => usePipelineStore.getState().setDrawerOpen(true), 600);

    } catch (err: any) {
      timers.forEach(clearTimeout);
      usePipelineStore.getState().setError(err.message || 'Failed to connect');
      usePipelineStore.getState().setRunState('error');
      usePipelineStore.getState().activateStage(0);
    }
  }, []);

  const stop = useCallback(() => {
    wsRef.current?.close();
    usePipelineStore.getState().setRunState('idle');
    usePipelineStore.getState().setAutoStream(false);
  }, []);

  // Auto-stream
  const autoStream = usePipelineStore((s) => s.autoStream);
  const runState = usePipelineStore((s) => s.runState);
  const interval = usePipelineStore((s) => s.autoStreamInterval);

  useEffect(() => {
    if (!autoStream) return;
    if (runState !== 'idle' && runState !== 'complete') return;

    const timer = setTimeout(() => {
      run();
    }, runState === 'complete' ? interval : 0);

    return () => clearTimeout(timer);
  }, [autoStream, runState, interval, run]);

  return { run, stop };
}
