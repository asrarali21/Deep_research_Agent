"use client";

import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo } from "react";

import { type ApiError, researchApi } from "@/features/research/api/client";
import { useReducedMotionSafe } from "@/features/research/hooks/use-reduced-motion-safe";
import { useResearchSessionStore } from "@/features/research/store/research-session-store";
import { buildResumeFeedback } from "@/features/research/utils/build-resume-feedback";
import { clearSessionSnapshot, loadSessionSnapshot, saveSessionSnapshot } from "@/features/research/utils/persistence";

function getRetryTimestamp(seconds?: number) {
  if (!seconds) {
    return undefined;
  }
  return Date.now() + seconds * 1000;
}

export function useResearchSession() {
  const reducedMotion = useReducedMotionSafe();
  const state = useResearchSessionStore();

  const hydrate = useCallback(() => {
    const snapshot = loadSessionSnapshot();
    if (snapshot) {
      state.hydrateFromStorage(snapshot);
      state.setRecoveryBanner(Boolean(snapshot.threadId && !["done", "failed"].includes(snapshot.status)));
    }
  }, [state]);

  useEffect(() => {
    hydrate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    saveSessionSnapshot({
      query: state.query,
      threadId: state.threadId,
      status: state.status,
      timeline: state.timeline,
      plan: state.plan,
      editablePlan: state.editablePlan,
      requiredSections: state.requiredSections,
      rawReport: state.rawReport,
      visibleReport: state.visibleReport,
      sources: state.sources,
      queuePosition: state.queuePosition,
      retryCount: state.retryCount,
      providerSwitchCount: state.providerSwitchCount,
      extractedFactsCount: state.extractedFactsCount,
      evidenceCardCount: state.evidenceCardCount,
      error: state.error,
      rateLimitResetAt: state.rateLimitResetAt,
      queueRetryAt: state.queueRetryAt,
      isStreaming: state.isStreaming,
      isThinkingOpen: state.isThinkingOpen,
      isSourcesOpenMobile: state.isSourcesOpenMobile,
      connectionLost: state.connectionLost,
      stoppedByUser: state.stoppedByUser,
      showRecoveryBanner: state.showRecoveryBanner,
    });
  }, [
    state.query,
    state.threadId,
    state.status,
    state.timeline,
    state.plan,
    state.editablePlan,
    state.requiredSections,
    state.rawReport,
    state.visibleReport,
    state.sources,
    state.queuePosition,
    state.retryCount,
    state.providerSwitchCount,
    state.extractedFactsCount,
    state.evidenceCardCount,
    state.error,
    state.rateLimitResetAt,
    state.queueRetryAt,
    state.isStreaming,
    state.isThinkingOpen,
    state.isSourcesOpenMobile,
    state.connectionLost,
    state.stoppedByUser,
    state.showRecoveryBanner,
  ]);

  const statusQuery = useQuery({
    queryKey: ["research-status", state.threadId],
    queryFn: () => researchApi.fetchStatus(state.threadId!),
    enabled:
      Boolean(state.threadId) &&
      !state.isStreaming &&
      ["queued", "running", "paused", "stopped"].includes(state.status),
    refetchInterval: 2_000,
    retry: false,
  });

  useEffect(() => {
    if (!statusQuery.data) {
      return;
    }
    useResearchSessionStore.getState().setStatusFromPoll(statusQuery.data);
    if (statusQuery.data.status === "done" || statusQuery.data.status === "failed") {
      useResearchSessionStore.getState().setRecoveryBanner(false);
    }
  }, [statusQuery.data]);

  useEffect(() => {
    const report = state.rawReport;
    if (!report) {
      return;
    }

    if (reducedMotion) {
      useResearchSessionStore.getState().setVisibleReport(report);
      return;
    }

    let cancelled = false;
    const chunkSize = 36;

    const tick = () => {
      if (cancelled) {
        return;
      }

      const snapshot = useResearchSessionStore.getState();
      const currentLength = snapshot.visibleReport.length;
      const nextLength = Math.min(snapshot.rawReport.length, currentLength + chunkSize);

      snapshot.setVisibleReport(snapshot.rawReport.slice(0, nextLength));

      if (nextLength < snapshot.rawReport.length) {
        requestAnimationFrame(tick);
      }
    };

    requestAnimationFrame(tick);
    return () => {
      cancelled = true;
    };
  }, [reducedMotion, state.rawReport]);

  const handleStreamFailure = useCallback(
    (error: unknown) => {
      if ((error as DOMException)?.name === "AbortError") {
        return;
      }

      const apiError = error as ApiError;
      state.markStreamResolved();
      state.setConnectionLost(true);
      state.setRecoveryBanner(Boolean(state.threadId));

      if (apiError?.status === 429) {
        useResearchSessionStore.setState({
          status: "failed",
          error: apiError.message,
          rateLimitResetAt: getRetryTimestamp(apiError.retryAfterSeconds),
          queueRetryAt: undefined,
        });
        return;
      }

      if (apiError?.status === 503) {
        useResearchSessionStore.setState({
          status: "failed",
          error: apiError.message,
          queueRetryAt: getRetryTimestamp(apiError.retryAfterSeconds),
          rateLimitResetAt: undefined,
        });
        return;
      }

      if (!state.rawReport) {
        useResearchSessionStore.setState({
          status: state.stoppedByUser ? "stopped" : "failed",
          error: apiError?.message ?? "The stream ended unexpectedly.",
        });
      }
    },
    [state],
  );

  const attachStream = useCallback(
    async (runner: (signal: AbortSignal) => Promise<void>) => {
      const controller = new AbortController();
      state.setStreamController(controller);
      state.setConnectionLost(false);
      useResearchSessionStore.setState({ isStreaming: true, error: undefined, rateLimitResetAt: undefined, queueRetryAt: undefined });

      try {
        await runner(controller.signal);
        state.markStreamResolved();
        const snapshot = useResearchSessionStore.getState();
        if (!["paused", "done", "failed", "stopped"].includes(snapshot.status)) {
          snapshot.setConnectionLost(true);
          snapshot.setRecoveryBanner(Boolean(snapshot.threadId));
        }
      } catch (error) {
        handleStreamFailure(error);
      }
    },
    [handleStreamFailure, state],
  );

  const startResearch = useCallback(
    async (query: string) => {
      state.startSession(query);
      clearSessionSnapshot();
      await attachStream((signal) => researchApi.startStream(query, state.applyEvent, signal));
    },
    [attachStream, state],
  );

  const resumeResearch = useCallback(
    async (note: string) => {
      if (!state.threadId) {
        return;
      }

      const feedback = buildResumeFeedback(state.plan, state.editablePlan, note);
      await attachStream((signal) => researchApi.resumeStream(state.threadId!, feedback, state.applyEvent, signal));
    },
    [attachStream, state],
  );

  const continueWithoutChanges = useCallback(async () => {
    if (!state.threadId) {
      return;
    }
    await attachStream((signal) => researchApi.resumeStream(state.threadId!, "", state.applyEvent, signal));
  }, [attachStream, state]);

  const stopStream = useCallback(() => {
    state.streamController?.abort();
    state.markStoppedByUser();
    state.setRecoveryBanner(Boolean(state.threadId));
  }, [state]);

  const resetSession = useCallback(() => {
    state.streamController?.abort();
    state.reset();
    clearSessionSnapshot();
  }, [state]);

  const canSubmit = useMemo(() => !state.isStreaming && state.status !== "queued" && state.status !== "running", [state]);

  return {
    state,
    statusQuery,
    actions: {
      startResearch,
      resumeResearch,
      continueWithoutChanges,
      stopStream,
      resetSession,
      setEditablePlan: state.setEditablePlan,
      setThinkingOpen: state.setThinkingOpen,
      setSourcesOpenMobile: state.setSourcesOpenMobile,
      clearError: state.clearError,
    },
    canSubmit,
  };
}
