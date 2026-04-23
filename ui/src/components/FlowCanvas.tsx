import { ReactFlow, Background, type NodeTypes, type EdgeTypes } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { initialNodes, initialEdges } from '../constants/layout';
import { ClientRequestNode } from './nodes/ClientRequestNode';
import { SchemaClassifierNode } from './nodes/SchemaClassifierNode';
import { TwinGeneratorNode } from './nodes/TwinGeneratorNode';
import { UpstreamNode } from './nodes/UpstreamNode';
import { FairnessEvaluatorNode } from './nodes/FairnessEvaluatorNode';
import { RLAgentNode } from './nodes/RLAgentNode';
import { VerdictNode } from './nodes/VerdictNode';
import { AuditLogNode } from './nodes/AuditLogNode';
import { AnimatedEdge } from './edges/AnimatedEdge';

const nodeTypes: NodeTypes = {
  clientRequest: ClientRequestNode,
  schemaClassifier: SchemaClassifierNode,
  twinGenerator: TwinGeneratorNode,
  upstreamAi: UpstreamNode,
  fairnessEvaluator: FairnessEvaluatorNode,
  rlAgent: RLAgentNode,
  verdict: VerdictNode,
  auditLog: AuditLogNode,
};

const edgeTypes: EdgeTypes = {
  animatedEdge: AnimatedEdge,
};

export function FlowCanvas() {
  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={initialNodes}
        edges={initialEdges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={true}
        zoomOnScroll={true}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.3}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#1a1f2a" gap={40} size={1} />
      </ReactFlow>
    </div>
  );
}
