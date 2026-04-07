export type SessionStatus =
  | "idle"
  | "queued"
  | "running"
  | "paused"
  | "done"
  | "failed"
  | "stopped";

export type ResearchStatus = {
  thread_id: string;
  status: "queued" | "running" | "paused" | "done" | "failed";
  queue_position: number | null;
  retry_count: number;
  provider_switch_count: number;
  current_plan: string[];
  extracted_facts_count: number;
  last_error: string;
};

export type QueuedEvent = {
  event: "queued";
  data: { thread_id: string; status: "queued"; resume?: true };
};

export type StartedEvent = {
  event: "started";
  data: { thread_id: string; status: "running"; worker: number };
};

export type PlanEvent = {
  event: "plan";
  data: { thread_id: string; task_count: number; plan: string[] };
};

export type AgentEvent = {
  event: "agent";
  data: { thread_id: string; status: "sub_agent_complete" };
};

export type EvaluateEvent = {
  event: "evaluate";
  data: { thread_id: string; gaps: string[]; gap_count: number };
};

export type SynthesizeEvent = {
  event: "synthesize";
  data: { thread_id: string; status: "finalizing_report" };
};

export type ReportEvent = {
  event: "report";
  data: { thread_id: string; report: string };
};

export type PausedEvent = {
  event: "paused";
  data: { thread_id: string; status: "paused" };
};

export type DoneEvent = {
  event: "done";
  data: { thread_id: string; status: "done" };
};

export type FailedEvent = {
  event: "failed";
  data: { thread_id: string; status: "failed"; error: string };
};

export type HeartbeatEvent = {
  event: "heartbeat";
  data: ResearchStatus;
};

export type ResearchEvent =
  | QueuedEvent
  | StartedEvent
  | PlanEvent
  | AgentEvent
  | EvaluateEvent
  | SynthesizeEvent
  | ReportEvent
  | PausedEvent
  | DoneEvent
  | FailedEvent
  | HeartbeatEvent;

export type TimelineItem = {
  id: string;
  kind: "queued" | "plan" | "research" | "evaluate" | "synthesize" | "system" | "error";
  title: string;
  description?: string;
  meta?: string[];
  createdAt: number;
};

export type SourceCard = {
  id: string;
  index: number;
  title: string;
  url: string;
  hostname: string;
  note?: string;
};

export type ResearchSessionState = {
  query: string;
  threadId?: string;
  status: SessionStatus;
  timeline: TimelineItem[];
  plan: string[];
  editablePlan: string[];
  rawReport: string;
  visibleReport: string;
  sources: SourceCard[];
  queuePosition?: number | null;
  retryCount: number;
  providerSwitchCount: number;
  extractedFactsCount: number;
  error?: string;
  rateLimitResetAt?: number;
  queueRetryAt?: number;
  isStreaming: boolean;
  isThinkingOpen: boolean;
  isSourcesOpenMobile: boolean;
  connectionLost: boolean;
  stoppedByUser: boolean;
  showRecoveryBanner: boolean;
};

export type SessionSnapshot = Omit<ResearchSessionState, "isStreaming">;
