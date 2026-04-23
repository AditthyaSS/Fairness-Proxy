import { useEffect, useCallback } from 'react';
import { usePipelineStore } from '../store/usePipelineStore';

export function useAuditStats() {
  const baseUrl = usePipelineStore((s) => s.baseUrl);
  const setAuditStats = usePipelineStore((s) => s.setAuditStats);
  const setApiOnline = usePipelineStore((s) => s.setApiOnline);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${baseUrl}/v1/audit/stats`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const data = await res.json();
        setAuditStats(data);
        setApiOnline(true);
      } else {
        setApiOnline(false);
      }
    } catch {
      setApiOnline(false);
    }
  }, [baseUrl, setAuditStats, setApiOnline]);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 15000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  return { refetch: fetchStats };
}
