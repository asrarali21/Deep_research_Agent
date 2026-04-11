"use client";

import { type RefObject } from "react";

import { ScrollArea } from "@/components/ui/scroll-area";
import { AIResponseRenderer } from "@/features/research/components/ai-response-renderer";
import { EmptyState } from "@/features/research/components/empty-state";
import { ErrorState } from "@/features/research/components/error-state";
import { MessageBubble } from "@/features/research/components/message-bubble";
import { PlanReviewPanel } from "@/features/research/components/plan-review-panel";
import { ScrollToBottom } from "@/features/research/components/scroll-to-bottom";
import { ThinkingSection } from "@/features/research/components/thinking-section";
import type { ResearchSessionState } from "@/features/research/types/research";

type MessageListProps = {
  session: ResearchSessionState;
  containerRef: RefObject<HTMLDivElement | null>;
  showScrollToBottom: boolean;
  onScrollToBottom: () => void;
  onExampleSelect: (value: string) => void;
  onPlanChange: (plan: string[]) => void;
  onResume: (note: string) => void;
  onContinue: () => void;
  onRetry: () => void;
  onRegenerate: () => void;
  onToggleThinking: () => void;
};

export function MessageList({
  session,
  containerRef,
  showScrollToBottom,
  onScrollToBottom,
  onExampleSelect,
  onPlanChange,
  onResume,
  onContinue,
  onRetry,
  onRegenerate,
  onToggleThinking,
}: MessageListProps) {
  const shouldShowReport =
    Boolean(session.rawReport) ||
    session.timeline.some((item) => item.kind === "synthesize") ||
    session.status === "done";

  return (
    <ScrollArea ref={containerRef} className="h-[calc(100vh-280px)] pr-2">
      <div className="space-y-5 pb-10">
        {!session.query ? <EmptyState onExampleSelect={onExampleSelect} /> : null}

        {session.query ? (
          <MessageBubble variant="user" title="Research brief">
            {session.query}
          </MessageBubble>
        ) : null}

        {(session.timeline.length > 0 || session.status !== "idle") && session.query ? (
          <ThinkingSection
            items={session.timeline}
            status={session.status}
            open={session.isThinkingOpen}
            onToggle={onToggleThinking}
          />
        ) : null}

        {session.showRecoveryBanner && session.threadId ? (
          <MessageBubble variant="system" title="Recovered session">
            This browser restored your last known session snapshot. If the backend job already finished elsewhere, the report is only available if it was cached in this browser.
          </MessageBubble>
        ) : null}

        {session.stoppedByUser ? (
          <MessageBubble variant="system" title="Stream stopped">
            Live streaming was stopped in this browser. The backend job may still continue on the server.
          </MessageBubble>
        ) : null}

        {session.connectionLost && !session.isStreaming ? (
          <MessageBubble variant="system" title="Connection recovery">
            The live stream disconnected, so the UI switched to status polling while it waits for a terminal state.
          </MessageBubble>
        ) : null}

        {session.status === "waiting_for_quota" ? (
          <MessageBubble variant="system" title="Waiting for provider quota">
            The backend paused this run while provider quota resets.
            {session.waitingTaskType ? ` The blocked stage is ${session.waitingTaskType}.` : ""}
            {session.quotaWaitUntil ? " It will resume automatically once quota becomes available." : ""}
          </MessageBubble>
        ) : null}

        {session.status === "paused" ? (
          <PlanReviewPanel
            plan={session.editablePlan}
            onChange={onPlanChange}
            onResume={onResume}
            onContinue={onContinue}
          />
        ) : null}

        {shouldShowReport ? (
          <AIResponseRenderer
            markdown={session.visibleReport || session.rawReport}
            isStreamingReveal={Boolean(session.rawReport && session.visibleReport !== session.rawReport)}
          />
        ) : null}

        {session.status === "failed" && session.error ? (
          <ErrorState
            error={session.error}
            onRetry={onRetry}
            onRegenerate={onRegenerate}
            rateLimitResetAt={session.rateLimitResetAt}
            queueRetryAt={session.queueRetryAt}
          />
        ) : null}
      </div>

      <ScrollToBottom visible={showScrollToBottom} onClick={onScrollToBottom} />
    </ScrollArea>
  );
}
