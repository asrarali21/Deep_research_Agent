"use client";

import { create } from "zustand";

import type { ResearchEvent, ResearchSessionState, ResearchStatus, SessionSnapshot } from "@/features/research/types/research";
import { mapEventToTimeline } from "@/features/research/utils/map-event-to-timeline";
import { parseSourcesFromMarkdown } from "@/features/research/utils/parse-sources";

function sourceCardFromUrl(url: string, note?: string) {
  try {
    const parsed = new URL(url);
    const hostname = parsed.hostname.replace(/^www\./, "");
    return {
      id: url,
      index: 0,
      title: hostname,
      url,
      hostname,
      note,
    };
  } catch {
    return null;
  }
}

function mergeSources(existing: ResearchSessionState["sources"], urls: string[], note?: string) {
  const merged = new Map(existing.map((source) => [source.url, source]));
  for (const url of urls) {
    const next = sourceCardFromUrl(url, note);
    if (!next) {
      continue;
    }
    const previous = merged.get(url);
    merged.set(url, previous ? { ...previous, note: previous.note || note } : next);
  }

  return Array.from(merged.values()).map((source, index) => ({
    ...source,
    index: index + 1,
  }));
}

type ResearchSessionActions = {
  startSession: (query: string) => void;
  reset: () => void;
  applyEvent: (event: ResearchEvent) => void;
  setStatusFromPoll: (status: ResearchStatus) => void;
  setVisibleReport: (value: string) => void;
  setEditablePlan: (plan: string[]) => void;
  setThinkingOpen: (open: boolean) => void;
  setSourcesOpenMobile: (open: boolean) => void;
  setStreamController: (controller: AbortController | null) => void;
  setConnectionLost: (lost: boolean) => void;
  hydrateFromStorage: (snapshot: SessionSnapshot) => void;
  markStreamResolved: () => void;
  markStoppedByUser: () => void;
  setRecoveryBanner: (visible: boolean) => void;
  clearError: () => void;
};

export type ResearchSessionStore = ResearchSessionState &
  ResearchSessionActions & {
    streamController: AbortController | null;
  };

export const initialSessionState: ResearchSessionState = {
  query: "",
  threadId: undefined,
  status: "idle",
  timeline: [],
  plan: [],
  editablePlan: [],
  requiredSections: [],
  rawReport: "",
  visibleReport: "",
  sources: [],
  queuePosition: null,
  retryCount: 0,
  providerSwitchCount: 0,
  extractedFactsCount: 0,
  evidenceCardCount: 0,
  error: undefined,
  rateLimitResetAt: undefined,
  queueRetryAt: undefined,
  isStreaming: false,
  isThinkingOpen: true,
  isSourcesOpenMobile: false,
  connectionLost: false,
  stoppedByUser: false,
  showRecoveryBanner: false,
};

export const useResearchSessionStore = create<ResearchSessionStore>((set) => ({
  ...initialSessionState,
  streamController: null,
  startSession: (query) =>
    set(() => ({
      ...initialSessionState,
      query,
      isStreaming: true,
      status: "queued",
      timeline: [],
    })),
  reset: () => set(() => ({ ...initialSessionState, streamController: null })),
  applyEvent: (event) =>
    set((state) => {
      const nextTimelineItem = mapEventToTimeline(event);
      const timeline = nextTimelineItem ? [...state.timeline, nextTimelineItem] : state.timeline;

      switch (event.event) {
        case "queued":
          return {
            ...state,
            status: "queued",
            threadId: event.data.thread_id,
            timeline,
            error: undefined,
            connectionLost: false,
            stoppedByUser: false,
            showRecoveryBanner: false,
          };
        case "started":
          return {
            ...state,
            status: "running",
            timeline,
            error: undefined,
          };
        case "plan":
          return {
            ...state,
            timeline,
            plan: event.data.plan,
            editablePlan: event.data.plan,
            requiredSections: event.data.required_sections?.length ? event.data.required_sections : state.requiredSections,
          };
        case "source_batch":
          return {
            ...state,
            status: state.status === "queued" ? "running" : state.status,
            timeline,
            sources: mergeSources(
              state.sources,
              [...event.data.discovered_sources, ...event.data.scraped_sources],
              event.data.task ? `Observed during: ${event.data.task}` : "Observed during research",
            ),
          };
        case "evidence_batch":
          return {
            ...state,
            status: state.status === "queued" ? "running" : state.status,
            timeline,
            evidenceCardCount: Math.max(state.evidenceCardCount, event.data.evidence_count),
            extractedFactsCount: Math.max(state.extractedFactsCount, event.data.finding_count),
          };
        case "agent":
        case "outline":
        case "section_draft":
        case "evaluate":
        case "synthesize":
          return {
            ...state,
            status: state.status === "queued" ? "running" : state.status,
            timeline,
          };
        case "report": {
          const parsedSources = parseSourcesFromMarkdown(event.data.report);
          return {
            ...state,
            rawReport: event.data.report,
            sources: mergeSources(
              state.sources,
              parsedSources.map((source) => source.url),
              "Referenced in final report",
            ).map((source) => parsedSources.find((parsed) => parsed.url === source.url) ?? source),
            visibleReport: "",
            timeline,
          };
        }
        case "paused":
          return {
            ...state,
            status: "paused",
            isStreaming: false,
            timeline,
          };
        case "done":
          return {
            ...state,
            status: "done",
            isStreaming: false,
            connectionLost: false,
            timeline,
          };
        case "failed":
          return {
            ...state,
            status: "failed",
            error: event.data.error,
            isStreaming: false,
            connectionLost: false,
            timeline,
          };
        case "heartbeat":
          return {
            ...state,
            status: state.stoppedByUser ? "stopped" : event.data.status,
            threadId: event.data.thread_id,
            queuePosition: event.data.queue_position,
            retryCount: event.data.retry_count,
            providerSwitchCount: event.data.provider_switch_count,
            extractedFactsCount: event.data.extracted_facts_count,
            evidenceCardCount: event.data.evidence_card_count ?? state.evidenceCardCount,
            plan: event.data.current_plan.length ? event.data.current_plan : state.plan,
            editablePlan: event.data.current_plan.length ? event.data.current_plan : state.editablePlan,
            requiredSections: event.data.required_sections?.length ? event.data.required_sections : state.requiredSections,
            error: event.data.last_error || state.error,
          };
      }
    }),
  setStatusFromPoll: (status) =>
    set((state) => ({
      ...state,
      threadId: status.thread_id,
      status: state.stoppedByUser ? "stopped" : status.status,
      queuePosition: status.queue_position,
      retryCount: status.retry_count,
      providerSwitchCount: status.provider_switch_count,
      extractedFactsCount: status.extracted_facts_count,
      evidenceCardCount: status.evidence_card_count ?? state.evidenceCardCount,
      plan: status.current_plan.length ? status.current_plan : state.plan,
      editablePlan: status.current_plan.length ? status.current_plan : state.editablePlan,
      requiredSections: status.required_sections?.length ? status.required_sections : state.requiredSections,
      error: status.last_error || state.error,
    })),
  setVisibleReport: (value) => set((state) => ({ ...state, visibleReport: value })),
  setEditablePlan: (plan) => set((state) => ({ ...state, editablePlan: plan })),
  setThinkingOpen: (open) => set((state) => ({ ...state, isThinkingOpen: open })),
  setSourcesOpenMobile: (open) => set((state) => ({ ...state, isSourcesOpenMobile: open })),
  setStreamController: (controller) => set((state) => ({ ...state, streamController: controller })),
  setConnectionLost: (lost) => set((state) => ({ ...state, connectionLost: lost })),
  hydrateFromStorage: (snapshot) =>
    set(() => ({
      ...initialSessionState,
      ...snapshot,
      isStreaming: false,
      streamController: null,
    })),
  markStreamResolved: () => set((state) => ({ ...state, isStreaming: false, streamController: null })),
  markStoppedByUser: () =>
    set((state) => ({
      ...state,
      status: "stopped",
      isStreaming: false,
      stoppedByUser: true,
      streamController: null,
    })),
  setRecoveryBanner: (visible) => set((state) => ({ ...state, showRecoveryBanner: visible })),
  clearError: () => set((state) => ({ ...state, error: undefined })),
}));
