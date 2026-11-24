export type BeliefType =
  | "static_surface"
  | "dynamic_state"
  | "workflow"
  | "gotcha"
  | "durable_rule"
  | "temporary";

export type BeliefStatus = "active" | "superseded" | "refuted" | "forgotten";

export type RevisionOp =
  | "ADD"
  | "SUPERSEDE"
  | "CONSOLIDATE"
  | "REFUTE"
  | "IGNORE"
  | "FORGET";

export interface Provenance {
  source_id: string;
  step_index: number | null;
  url?: string | null;
  outcome: "success" | "failure" | "unknown";
  excerpt?: string | null;
}

export interface Belief {
  id: string;
  type: BeliefType;
  surface: string;
  claim: string;
  closed_set_items: string[];
  status: BeliefStatus;
  confidence: number;
  durability: number;
  evidence_count: number;
  created_at_step: number;
  updated_at_step: number;
  superseded_by: string | null;
  provenance: Provenance[];
}

export interface BeliefEdge {
  source_id: string;
  target_id: string;
  relation: "SUPERSEDES" | "CONSOLIDATES" | "REFUTES";
  step: number;
  reason: string;
}

export interface TimelineEvent {
  id?: number | null;
  step: number;
  source_id: string;
  op: RevisionOp;
  belief_id?: string | null;
  target_belief_id?: string | null;
  summary: string;
}

export interface ReplayQuestionComparison {
  question_id: string;
  category: string;
  question: string;
  gold_answer: string;
  baseline_policy: string;
  baseline_answer: string;
  baseline_correct: boolean;
  baseline_retrieved: string[];
  proposed_policy: string;
  proposed_answer: string;
  proposed_correct: boolean;
  proposed_retrieved_belief_ids: string[];
}

export interface ReplayBundle {
  schema_version: string;
  run_id: string;
  benchmark: string;
  policy_name: string;
  total_steps: number;
  beliefs: Belief[];
  edges: BeliefEdge[];
  events: TimelineEvent[];
  comparisons: ReplayQuestionComparison[];
}
