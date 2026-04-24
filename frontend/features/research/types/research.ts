export type SessionStatus =
  | "idle"
  | "queued"
  | "running"
  | "waiting_for_quota"
  | "paused"
  | "done"
  | "failed"
  | "stopped";

export type ResearchStatus = {
  thread_id: string;
  status: "queued" | "running" | "waiting_for_quota" | "paused" | "done" | "failed";
  queue_position: number | null;
  retry_count: number;
  provider_switch_count: number;
  current_plan: string[];
  required_sections?: string[];
  extracted_facts_count: number;
  evidence_card_count?: number;
  quota_wait_until?: number | null;
  quota_retry_after_seconds?: number;
  waiting_task_type?: string;
  last_error: string;
};

export type QueuedEvent = {
  event: "queued";
  data: { thread_id: string; status: "queued"; resume?: true; reason?: string };
};

export type StartedEvent = {
  event: "started";
  data: { thread_id: string; status: "running"; worker: number };
};

export type WaitingForQuotaEvent = {
  event: "waiting_for_quota";
  data: {
    thread_id: string;
    status: "waiting_for_quota";
    task_type: string;
    retry_after_seconds: number;
    available_at: number;
    error: string;
  };
};

export type PlanEvent = {
  event: "plan";
  data: { thread_id: string; task_count: number; plan: string[]; required_sections?: string[] };
};

export type AgentEvent = {
  event: "agent";
  data: { thread_id: string; status: "sub_agent_complete"; task?: string; evidence_count?: number; source_count?: number };
};

export type SourceBatchEvent = {
  event: "source_batch";
  data: {
    thread_id: string;
    task: string;
    discovered_sources: string[];
    scraped_sources: string[];
    discovered_count: number;
    scraped_count: number;
  };
};

export type EvidenceBatchEvent = {
  event: "evidence_batch";
  data: {
    thread_id: string;
    task: string;
    evidence_count: number;
    finding_count: number;
    coverage_tags: string[];
  };
};

export type EvaluateEvent = {
  event: "evaluate";
  data: { thread_id: string; gaps: string[]; gap_count: number; quality_summary?: string };
};

export type OutlineEvent = {
  event: "outline";
  data: { thread_id: string; sections: string[] };
};

export type SectionDraftEvent = {
  event: "section_draft";
  data: { thread_id: string; section: string; status: "section_drafted"; char_count: number };
};

export type EvidenceBriefEvent = {
  event: "evidence_brief";
  data: { thread_id: string; section_count: number; priority_sections: string[] };
};

export type SectionVerificationEvent = {
  event: "section_verification";
  data: { thread_id: string; verified_sections: string[]; verification_count: number };
};

export type SynthesizeEvent = {
  event: "synthesize";
  data: { thread_id: string; status: "finalizing_report" | "final_editing_report" };
};

export type StructuredReference = {
  id: string;
  evidence_ids?: string[];
  title: string;
  url: string;
  hostname?: string;
  source_type?: string;
  verification_status?: string;
};

export type ReportEvent = {
  event: "report";
  data: { thread_id: string; report: string; structured_references?: StructuredReference[] };
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
  | WaitingForQuotaEvent
  | PlanEvent
  | AgentEvent
  | SourceBatchEvent
  | EvidenceBatchEvent
  | EvaluateEvent
  | OutlineEvent
  | EvidenceBriefEvent
  | SectionDraftEvent
  | SectionVerificationEvent
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
  referenceId?: string;
  evidenceIds?: string[];
  sourceType?: string;
  verificationStatus?: string;
};

export type ResearchSessionState = {
  query: string;
  threadId?: string;
  status: SessionStatus;
  timeline: TimelineItem[];
  plan: string[];
  editablePlan: string[];
  requiredSections: string[];
  rawReport: string;
  visibleReport: string;
  sources: SourceCard[];
  queuePosition?: number | null;
  retryCount: number;
  providerSwitchCount: number;
  extractedFactsCount: number;
  evidenceCardCount: number;
  error?: string;
  rateLimitResetAt?: number;
  queueRetryAt?: number;
  quotaWaitUntil?: number;
  waitingTaskType?: string;
  isStreaming: boolean;
  isThinkingOpen: boolean;
  isSourcesOpenMobile: boolean;
  connectionLost: boolean;
  stoppedByUser: boolean;
  showRecoveryBanner: boolean;
};

export type SessionSnapshot = Omit<ResearchSessionState, "isStreaming">;
