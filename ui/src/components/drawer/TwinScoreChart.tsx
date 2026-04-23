import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ReferenceLine, Tooltip, Cell } from 'recharts';
import { usePipelineStore } from '../../store/usePipelineStore';

export function TwinScoreChart() {
  const result = usePipelineStore((s) => s.result);

  if (!result?.fairness_metrics?.twin_scores) {
    return <div className="drawer-empty">Run the pipeline to see twin scores.</div>;
  }

  const originalScore = result.original_score;
  const twinScores = result.fairness_metrics.twin_scores;
  const meanTwin = twinScores.reduce((a, b) => a + b, 0) / twinScores.length;
  const lcf = result.fairness_metrics.counterfactual_variance;

  const data = [
    { x: 0, y: originalScore, label: 'Original', isOriginal: true },
    ...twinScores.map((s, i) => ({ x: i + 1, y: s, label: `Twin ${i + 1}`, isOriginal: false })),
  ];

  return (
    <div className="twin-chart">
      <div className="chart-legend">
        <span className="legend-item">
          <span className="legend-diamond" /> Original
        </span>
        <span className="legend-item">
          <span className="legend-dot" style={{ background: '#6b7280' }} /> Twins
        </span>
        <span className="legend-item" style={{ color: '#f5a623' }}>
          L_CF = {lcf.toFixed(6)}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <ScatterChart margin={{ left: 20, right: 20, top: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#242830" />
          <XAxis
            type="number"
            dataKey="x"
            domain={[-0.5, twinScores.length + 0.5]}
            tick={{ fontSize: 10, fill: '#6b7280' }}
            stroke="#242830"
          />
          <YAxis
            type="number"
            dataKey="y"
            domain={[0, 1]}
            tick={{ fontSize: 10, fill: '#6b7280', fontFamily: 'IBM Plex Mono' }}
            stroke="#242830"
          />
          <Tooltip
            contentStyle={{
              background: '#111318',
              border: '1px solid #242830',
              borderRadius: 6,
              fontSize: 11,
              color: '#e5e7eb',
            }}
            formatter={(value: number) => value.toFixed(4)}
          />
          <ReferenceLine y={meanTwin} stroke="#f5a623" strokeDasharray="4 4" label={{ value: `μ=${meanTwin.toFixed(3)}`, fill: '#f5a623', fontSize: 10 }} />
          <ReferenceLine y={originalScore} stroke="#e2e8f044" strokeDasharray="2 2" />
          <Scatter data={data} animationDuration={600}>
            {data.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.isOriginal ? '#e2e8f0' : '#64748b'}
                r={entry.isOriginal ? 6 : 5}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
