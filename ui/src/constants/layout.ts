import type { Node, Edge } from '@xyflow/react';

export const NODE_IDS = {
  CLIENT: 'client-request',
  SCHEMA: 'schema-classifier',
  TWINS: 'twin-generator',
  UPSTREAM: 'upstream-ai',
  FAIRNESS: 'fairness-evaluator',
  RL_AGENT: 'rl-agent',
  VERDICT: 'verdict',
  AUDIT: 'audit-log',
} as const;

const X_CENTER = 400;

export const initialNodes: Node[] = [
  {
    id: NODE_IDS.CLIENT,
    type: 'clientRequest',
    position: { x: X_CENTER - 140, y: 0 },
    data: {},
  },
  {
    id: NODE_IDS.SCHEMA,
    type: 'schemaClassifier',
    position: { x: X_CENTER - 170, y: 140 },
    data: {},
  },
  {
    id: NODE_IDS.TWINS,
    type: 'twinGenerator',
    position: { x: X_CENTER - 170, y: 310 },
    data: {},
  },
  {
    id: NODE_IDS.UPSTREAM,
    type: 'upstreamAi',
    position: { x: X_CENTER - 240, y: 480 },
    data: {},
  },
  {
    id: NODE_IDS.FAIRNESS,
    type: 'fairnessEvaluator',
    position: { x: X_CENTER - 200, y: 670 },
    data: {},
  },
  {
    id: NODE_IDS.RL_AGENT,
    type: 'rlAgent',
    position: { x: X_CENTER - 170, y: 870 },
    data: {},
  },
  {
    id: NODE_IDS.VERDICT,
    type: 'verdict',
    position: { x: X_CENTER - 220, y: 1050 },
    data: {},
  },
  {
    id: NODE_IDS.AUDIT,
    type: 'auditLog',
    position: { x: X_CENTER - 160, y: 1230 },
    data: {},
  },
];

export const initialEdges: Edge[] = [
  { id: 'e-client-schema', source: NODE_IDS.CLIENT, target: NODE_IDS.SCHEMA, type: 'animatedEdge' },
  { id: 'e-schema-twins', source: NODE_IDS.SCHEMA, target: NODE_IDS.TWINS, type: 'animatedEdge' },
  { id: 'e-twins-upstream', source: NODE_IDS.TWINS, target: NODE_IDS.UPSTREAM, type: 'animatedEdge' },
  { id: 'e-upstream-fairness', source: NODE_IDS.UPSTREAM, target: NODE_IDS.FAIRNESS, type: 'animatedEdge' },
  { id: 'e-fairness-rl', source: NODE_IDS.FAIRNESS, target: NODE_IDS.RL_AGENT, type: 'animatedEdge' },
  { id: 'e-rl-verdict', source: NODE_IDS.RL_AGENT, target: NODE_IDS.VERDICT, type: 'animatedEdge' },
  { id: 'e-verdict-audit', source: NODE_IDS.VERDICT, target: NODE_IDS.AUDIT, type: 'animatedEdge' },
];

export const STAGE_COLORS: Record<string, string> = {
  [NODE_IDS.CLIENT]: '#6b7280',
  [NODE_IDS.SCHEMA]: '#4f8ef7',
  [NODE_IDS.TWINS]: '#7c3aed',
  [NODE_IDS.UPSTREAM]: '#64748b',
  [NODE_IDS.FAIRNESS]: '#f5a623',
  [NODE_IDS.RL_AGENT]: '#6366f1',
  [NODE_IDS.VERDICT]: '#6b7280',
  [NODE_IDS.AUDIT]: '#6b7280',
};
