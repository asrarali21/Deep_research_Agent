"use client";

import { ArrowRight, RotateCcw } from "lucide-react";
import { useMemo, useRef, useState } from "react";

import { AppShell } from "@/components/layout/app-shell";
import { Button } from "@/components/ui/button";
import { useAutoScroll } from "@/features/research/hooks/use-auto-scroll";
import { useResearchSession } from "@/features/research/hooks/use-research-session";
import { InputBox } from "@/features/research/components/input-box";
import { MessageList } from "@/features/research/components/message-list";
import { Sidebar } from "@/features/research/components/sidebar";
import { SourcesPanel } from "@/features/research/components/sources-panel";
import { StatusStrip } from "@/features/research/components/status-strip";
import { Toolbar } from "@/features/research/components/toolbar";

export function ChatLayout() {
  const { state, actions, canSubmit } = useResearchSession();
  const [composerValue, setComposerValue] = useState("");
  const [planNoteValue, setPlanNoteValue] = useState("");
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const { isPinnedToBottom, scrollToBottom } = useAutoScroll(scrollRef, [
    state.timeline.length,
    state.visibleReport.length,
    state.status,
  ]);

  const inputMode = state.status === "paused" ? "resume" : "query";
  const inputValue = inputMode === "query" ? composerValue : planNoteValue;

  const submitLabel = useMemo(() => {
    if (inputMode === "resume") {
      return "Revise & continue";
    }
    return state.query ? "Start fresh run" : "Start research";
  }, [inputMode, state.query]);

  return (
    <AppShell
      sidebar={
        <Sidebar
          query={state.query}
          status={state.status}
          threadId={state.threadId}
          onNewQuery={() => {
            actions.resetSession();
            setComposerValue("");
            setPlanNoteValue("");
          }}
        />
      }
      main={
        <div className="flex h-full min-h-[calc(100vh-140px)] flex-col gap-4">
          <StatusStrip
            status={state.status}
            queuePosition={state.queuePosition}
            retryCount={state.retryCount}
            providerSwitchCount={state.providerSwitchCount}
            factsCount={state.extractedFactsCount}
            evidenceCount={state.evidenceCardCount}
          />

          <div className="flex flex-wrap items-center justify-between gap-3">
            <Toolbar
              canStop={state.isStreaming}
              canRetry={state.status === "failed" && Boolean(state.query)}
              canRegenerate={Boolean(state.query) && !state.isStreaming}
              report={state.rawReport}
              onStop={actions.stopStream}
              onRetry={() => {
                if (state.status === "failed" && state.query) {
                  setComposerValue(state.query);
                  actions.clearError();
                  void actions.startResearch(state.query);
                }
              }}
              onRegenerate={() => {
                if (state.query) {
                  setComposerValue(state.query);
                  actions.clearError();
                  void actions.startResearch(state.query);
                }
              }}
            />

            {state.status === "paused" ? (
              <Button variant="secondary" onClick={() => void actions.continueWithoutChanges()}>
                <ArrowRight className="h-4 w-4" />
                Continue current plan
              </Button>
            ) : state.query ? (
              <Button variant="ghost" onClick={() => actions.resetSession()}>
                <RotateCcw className="h-4 w-4" />
                Reset workspace
              </Button>
            ) : null}
          </div>

          <MessageList
            session={state}
            containerRef={scrollRef}
            showScrollToBottom={!isPinnedToBottom}
            onScrollToBottom={scrollToBottom}
            onExampleSelect={(value) => setComposerValue(value)}
            onPlanChange={actions.setEditablePlan}
            onResume={(note) => {
              setPlanNoteValue(note);
              void actions.resumeResearch(note);
            }}
            onContinue={() => void actions.continueWithoutChanges()}
            onRetry={() => {
              if (state.query) {
                void actions.startResearch(state.query);
              }
            }}
            onRegenerate={() => {
              if (state.query) {
                void actions.startResearch(state.query);
              }
            }}
            onToggleThinking={() => actions.setThinkingOpen(!state.isThinkingOpen)}
          />

          <InputBox
            mode={inputMode}
            value={inputValue}
            onChange={inputMode === "query" ? setComposerValue : setPlanNoteValue}
            onSubmit={() => {
              if (inputMode === "resume") {
                void actions.resumeResearch(planNoteValue);
                return;
              }
              if (composerValue.trim()) {
                void actions.startResearch(composerValue.trim());
              }
            }}
            disabled={inputMode === "query" ? !canSubmit : state.isStreaming}
            placeholder={
              inputMode === "query"
                ? "Ask for a deep research brief with scope, entities, or questions to investigate."
                : "Optional guidance for revising the plan before resuming."
            }
            submitLabel={submitLabel}
          />
        </div>
      }
      sources={
        <SourcesPanel
          sources={state.sources}
          status={state.status}
          isMobileOpen={state.isSourcesOpenMobile}
          onToggleMobile={actions.setSourcesOpenMobile}
        />
      }
    />
  );
}
