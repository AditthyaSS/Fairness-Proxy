export interface FeatureImportance {
  feature_name: string;
  shap_value: number;
  abs_shap_value: number;
  is_protected: boolean;
}

export interface XAIReport {
  feature_importances: FeatureImportance[];
  total_protected_shap: number;
  top_protected_feature: string;
}

export interface FairnessMetrics {
  counterfactual_variance: number;
  max_score_delta: number;
  mean_score_delta: number;
  demographic_parity_ratio: number | null;
  twin_scores: number[];
  rl_reward: number;
  xai_report: XAIReport;
}

export type Verdict = 'PASS' | 'MITIGATE' | 'BLOCK';

export interface RLDecision {
  verdict: Verdict;
  action_taken: string;
  corrected_score: number | null;
  explanation: string;
}

export interface ClassifiedSchema {
  domain: string;
  domain_confidence: number;
  merit_features: Record<string, any>;
  protected_features: Record<string, any>;
}

export interface Twin {
  twin_id: number;
  swapped_features: Record<string, any>;
}

export interface TwinGeneration {
  twins: Twin[];
}

export interface UpstreamInference {
  decision_score: number;
  binary_decision: boolean;
  latency_ms: number;
}

export interface ProxyResponse {
  request_id: string;
  domain: string;
  domain_confidence: number;
  original_decision: boolean;
  original_score: number;
  final_decision: boolean;
  final_score: number;
  fairness_metrics: FairnessMetrics;
  rl_decision: RLDecision;
  classified_schema: ClassifiedSchema;
  twin_generation: TwinGeneration;
  upstream_inferences: UpstreamInference[];
  total_latency_ms: number;
  mitigation_applied: boolean;
}

export interface AuditStats {
  total_requests: number;
  pass_count: number;
  mitigate_count: number;
  block_count: number;
  avg_latency_ms: number;
}

export type Scenario = string;
export type RunState = 'idle' | 'running' | 'complete' | 'error';
export type DrawerTab = 'json' | 'shap' | 'twins' | 'rl';
