import { useEffect, useCallback } from 'react';
import { usePipelineStore } from '../store/usePipelineStore';

export function useRLStats() {
  const baseUrl = usePipelineStore((s) => s.baseUrl);
  const setRlStats = usePipelineStore((s) => s.setRlStats);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch(`${baseUrl}/v1/rl/stats`, {
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        const data = await res.json();
        setRlStats(data);
      }
    } catch {
      /* ignore */
    }
  }, [baseUrl, setRlStats]);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  return { refresh };
}
