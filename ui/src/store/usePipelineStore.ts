import { create } from 'zustand';
import type { ProxyResponse, RunState, DrawerTab } from '../types/api';

interface StageData {
  classified?: {
    domain: string;
    confidence: number;
    merit_features: string[];
    protected_features: string[];
  };
  twins_generated?: {
    num_twins: number;
    swapped_features: Record<string, any>[];
  };
  upstream_scores?: Array<{
    index: number;
    label: string;
    score: number;
    decision: boolean;
  }>;
}

interface RLStats {
  total_episodes: number;
  avg_reward_last50: number;
  avg_cf_variance_last50: number;
  avg_protected_shap_last50: number;
  action_distribution_last50: Record<string, number>;
  reward_curve: number[];
  agent_mode: string;
  // Accuracy tracking
  original_accuracy_curve: number[];
  corrected_accuracy_curve: number[];
  accuracy_gain_curve: number[];
  cumulative_improvement: number;
  total_labeled_episodes: number;
  agent_phase: 'heuristic_fallback' | 'ppo_warming' | 'trained_ppo';
  phase_progress: number;
}

interface AuditStats {
  total_requests: number;
  pass_count: number;
  mitigate_count: number;
  block_count: number;
  avg_latency_ms: number;
}

interface PipelineStore {
  // Run state
  runState: RunState;
  activeStages: Set<number>;
  completedStages: Set<number>;
  stageData: StageData;
  result: ProxyResponse | null;
  error: string | null;

  // Accuracy tracking
  accuracyGain: number | null;
  cumulativeImprovement: number | null;

  // UI state
  drawerOpen: boolean;
  drawerTab: DrawerTab;
  scenario: string;
  formValues: Record<string, any>;
  mockEnabled: boolean;
  baseUrl: string;
  autoStream: boolean;
  autoStreamInterval: number;

  // Stats
  auditStats: AuditStats | null;
  rlStats: RLStats | null;
  apiOnline: boolean;

  // Actions
  setRunState: (s: RunState) => void;
  activateStage: (n: number) => void;
  completeStage: (n: number) => void;
  setStageData: (d: Partial<StageData>) => void;
  setResult: (r: ProxyResponse | null) => void;
  setError: (e: string | null) => void;
  setAccuracyGain: (n: number | null) => void;
  setCumulativeImprovement: (n: number | null) => void;
  setDrawerOpen: (open: boolean) => void;
  setDrawerTab: (tab: DrawerTab) => void;
  setScenario: (s: string) => void;
  setFormValues: (v: Record<string, any>) => void;
  updateField: (key: string, value: any) => void;
  setMockEnabled: (b: boolean) => void;
  setBaseUrl: (u: string) => void;
  setAutoStream: (b: boolean) => void;
  setAuditStats: (s: AuditStats | null) => void;
  setRlStats: (s: RLStats | null) => void;
  setApiOnline: (b: boolean) => void;
  resetPipeline: () => void;
}

export const usePipelineStore = create<PipelineStore>((set) => ({
  runState: 'idle',
  activeStages: new Set<number>(),
  completedStages: new Set<number>(),
  stageData: {},
  result: null,
  error: null,
  accuracyGain: null,
  cumulativeImprovement: null,
  drawerOpen: false,
  drawerTab: 'json',
  scenario: 'adult_income',
  formValues: {},
  mockEnabled: true,
  baseUrl: 'http://localhost:8000',
  autoStream: false,
  autoStreamInterval: 3000,
  auditStats: null,
  rlStats: null,
  apiOnline: false,

  setRunState: (runState) => set({ runState }),
  activateStage: (n) =>
    set((s) => ({ activeStages: new Set(s.activeStages).add(n) })),
  completeStage: (n) =>
    set((s) => ({ completedStages: new Set(s.completedStages).add(n) })),
  setStageData: (d) =>
    set((s) => ({ stageData: { ...s.stageData, ...d } })),
  setResult: (result) => set({ result }),
  setError: (error) => set({ error }),
  setAccuracyGain: (accuracyGain) => set({ accuracyGain }),
  setCumulativeImprovement: (cumulativeImprovement) => set({ cumulativeImprovement }),
  setDrawerOpen: (drawerOpen) => set({ drawerOpen }),
  setDrawerTab: (drawerTab) => set({ drawerTab }),
  setScenario: (scenario) => set({ scenario }),
  setFormValues: (formValues) => set({ formValues }),
  updateField: (key, value) =>
    set((s) => ({ formValues: { ...s.formValues, [key]: value } })),
  setMockEnabled: (mockEnabled) => set({ mockEnabled }),
  setBaseUrl: (baseUrl) => set({ baseUrl }),
  setAutoStream: (autoStream) => set({ autoStream }),
  setAuditStats: (auditStats) => set({ auditStats }),
  setRlStats: (rlStats) => set({ rlStats }),
  setApiOnline: (apiOnline) => set({ apiOnline }),
  resetPipeline: () =>
    set({
      runState: 'idle',
      activeStages: new Set<number>(),
      completedStages: new Set<number>(),
      stageData: {},
      result: null,
      error: null,
      accuracyGain: null,
      cumulativeImprovement: null,
      drawerOpen: false,
    }),
}));
