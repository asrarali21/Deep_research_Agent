import { ArrowRight, Compass, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { SectionCard } from "@/components/layout/section-card";

const examples = [
  "Analyze India's EV charging market in 2026: major players, pricing trends, and policy tailwinds.",
  "Compare OpenAI, Google, and Anthropic enterprise AI positioning using recent public sources.",
  "Research the current state of climate-focused semiconductor manufacturing incentives in the US and EU.",
];

type EmptyStateProps = {
  onExampleSelect: (value: string) => void;
};

export function EmptyState({ onExampleSelect }: EmptyStateProps) {
  return (
    <SectionCard className="overflow-hidden p-6 sm:p-8">
      <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div>
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent/10 px-3 py-1 text-xs uppercase tracking-[0.18em] text-accent">
            <Sparkles className="h-3.5 w-3.5" />
            Deep research mode
          </div>
          <h1 className="max-w-2xl font-display text-4xl font-semibold tracking-tight text-text sm:text-5xl">
            Turn a single prompt into a transparent, source-backed research report.
          </h1>
          <p className="mt-4 max-w-2xl text-base leading-8 text-muted">
            This workspace streams the research process, pauses for plan review, and assembles the final answer as a polished document with references.
          </p>
        </div>

        <div className="rounded-panel border border-white/10 bg-black/20 p-5">
          <div className="mb-4 flex items-center gap-2 text-sm font-medium text-text">
            <Compass className="h-4 w-4 text-accent" />
            Example research briefs
          </div>
          <div className="space-y-3">
            {examples.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => onExampleSelect(example)}
                className="group block w-full rounded-[18px] border border-white/10 bg-white/[0.03] p-4 text-left transition hover:border-accent/30 hover:bg-white/[0.05]"
              >
                <p className="text-sm text-text">{example}</p>
                <span className="mt-3 inline-flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted transition group-hover:text-accent">
                  Use this brief
                  <ArrowRight className="h-3.5 w-3.5" />
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </SectionCard>
  );
}
