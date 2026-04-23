import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Cell,
  ReferenceLine,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { usePipelineStore } from '../../store/usePipelineStore';

export function ShapChart() {
  const result = usePipelineStore((s) => s.result);

  if (!result?.fairness_metrics?.xai_report?.feature_importances) {
    return <div className="drawer-empty">Run the pipeline to see SHAP values.</div>;
  }

  const features = result.fairness_metrics.xai_report.feature_importances
    .slice()
    .sort((a, b) => b.abs_shap_value - a.abs_shap_value);

  const data = features.map((f) => ({
    name: f.feature_name,
    value: f.abs_shap_value,
    isProtected: f.is_protected,
  }));

  return (
    <div className="shap-chart">
      <div className="chart-legend">
        <span className="legend-item">
          <span className="legend-dot" style={{ background: '#4f8ef7' }} /> Merit
        </span>
        <span className="legend-item">
          <span className="legend-dot" style={{ background: '#f5a623' }} /> Protected
        </span>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical" margin={{ left: 80, right: 20, top: 5, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#242830" horizontal={false} />
          <XAxis type="number" stroke="#6b7280" tick={{ fontSize: 10, fill: '#6b7280' }} />
          <YAxis
            type="category"
            dataKey="name"
            stroke="#6b7280"
            tick={{ fontSize: 10, fill: '#9ca3af', fontFamily: 'IBM Plex Mono' }}
            width={75}
          />
          <Tooltip
            contentStyle={{
              background: '#111318',
              border: '1px solid #242830',
              borderRadius: 6,
              fontSize: 11,
              color: '#e5e7eb',
            }}
          />
          <ReferenceLine x={0.05} stroke="#f5a623" strokeDasharray="4 4" label={{ value: 'MITIGATE', fill: '#f5a623', fontSize: 9 }} />
          <ReferenceLine x={0.15} stroke="#ff4757" strokeDasharray="4 4" label={{ value: 'BLOCK', fill: '#ff4757', fontSize: 9 }} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} animationDuration={800}>
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.isProtected ? '#f5a623' : '#4f8ef7'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
