import { BaseEdge, getSmoothStepPath, type EdgeProps } from '@xyflow/react';
import { usePipelineStore } from '../../store/usePipelineStore';
import { NODE_IDS, STAGE_COLORS } from '../../constants/layout';

const NODE_ORDER = [
  NODE_IDS.CLIENT,
  NODE_IDS.SCHEMA,
  NODE_IDS.TWINS,
  NODE_IDS.UPSTREAM,
  NODE_IDS.FAIRNESS,
  NODE_IDS.RL_AGENT,
  NODE_IDS.VERDICT,
  NODE_IDS.AUDIT,
];

// Edges where bias "flows" — these turn red/amber when bias is detected
const BIAS_FLOW_EDGES = new Set([
  NODE_IDS.UPSTREAM,   // upstream → fairness
  NODE_IDS.FAIRNESS,   // fairness → rl
  NODE_IDS.RL_AGENT,   // rl → verdict
]);

export function AnimatedEdge(props: EdgeProps) {
  const { sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, source } = props;

  const runState = usePipelineStore((s) => s.runState);
  const completedStages = usePipelineStore((s) => s.completedStages);
  const activeStages = usePipelineStore((s) => s.activeStages);
  const result = usePipelineStore((s) => s.result);

  const sourceIdx = NODE_ORDER.indexOf(source as any);
  const isActive = activeStages.has(sourceIdx) || activeStages.has(sourceIdx + 1);
  const isCompleted = completedStages.has(sourceIdx);

  // Determine bias severity for this edge
  const verdict = result?.rl_decision?.verdict;
  const isBiasEdge = BIAS_FLOW_EDGES.has(source as any);
  const hasBias = isCompleted && isBiasEdge && verdict && verdict !== 'PASS';
  const biasLevel = verdict === 'BLOCK' ? 'severe' : verdict === 'MITIGATE' ? 'moderate' : null;

  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 16,
  });

  // Bias-aware edge coloring
  let strokeColor: string;
  if (hasBias && biasLevel === 'severe') {
    strokeColor = '#ff4757';
  } else if (hasBias && biasLevel === 'moderate') {
    strokeColor = '#f5a623';
  } else if (isCompleted) {
    strokeColor = STAGE_COLORS[source] || '#4f8ef7';
  } else if (isActive) {
    strokeColor = '#4f8ef7';
  } else {
    strokeColor = '#242830';
  }

  const strokeWidth = hasBias ? 3 : (isActive || isCompleted ? 2 : 1);

  return (
    <>
      {/* Glow layer for bias edges */}
      {hasBias && (
        <BaseEdge
          path={edgePath}
          style={{
            stroke: strokeColor,
            strokeWidth: 8,
            strokeOpacity: 0.15,
            filter: `drop-shadow(0 0 4px ${strokeColor})`,
          }}
        />
      )}

      {/* Main edge */}
      <BaseEdge
        path={edgePath}
        style={{
          stroke: strokeColor,
          strokeWidth,
          strokeDasharray: isCompleted ? 'none' : '6 4',
          transition: 'stroke 0.3s ease, stroke-width 0.3s ease',
        }}
      />

      {/* Animated particles */}
      {isActive && !isCompleted && (
        <circle r="3" fill={strokeColor}>
          <animateMotion dur="1.5s" repeatCount="indefinite" path={edgePath} />
        </circle>
      )}

      {/* Bias warning particles — red/amber dots flowing on bias edges */}
      {hasBias && (
        <>
          <circle r="3" fill={strokeColor} opacity="0.9">
            <animateMotion dur="2s" repeatCount="indefinite" path={edgePath} />
          </circle>
          <circle r="2" fill={strokeColor} opacity="0.6">
            <animateMotion dur="2s" begin="0.7s" repeatCount="indefinite" path={edgePath} />
          </circle>
          <circle r="2" fill={strokeColor} opacity="0.4">
            <animateMotion dur="2s" begin="1.4s" repeatCount="indefinite" path={edgePath} />
          </circle>
        </>
      )}
    </>
  );
}
