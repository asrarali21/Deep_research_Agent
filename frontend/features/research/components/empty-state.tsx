import { ArrowRight, Compass, Sparkles } from "lucide-react";

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
    <SectionCard className="overflow-hidden">
      <div className="grid lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="p-6 sm:p-8">
          <div className="mb-5 inline-flex items-center gap-2 text-sm font-medium text-accent">
            <Sparkles className="h-3.5 w-3.5" />
            Deep research
          </div>
          <h1 className="max-w-2xl font-display text-3xl font-semibold text-text sm:text-4xl">
            Start with a question worth checking.
          </h1>
          <p className="mt-4 max-w-xl text-base leading-8 text-muted">
            Markets, policies, competitors, technical shifts, and messy strategic calls are all fair game.
          </p>
        </div>

        <div className="border-t border-white/[0.08] bg-black/10 lg:border-l lg:border-t-0">
          <div className="flex items-center gap-2 border-b border-white/[0.08] p-4 text-sm font-medium text-text">
            <Compass className="h-4 w-4 text-accent" />
            Example research briefs
          </div>
          <div className="divide-y divide-white/[0.08]">
            {examples.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => onExampleSelect(example)}
                className="group block w-full p-4 text-left transition hover:bg-white/[0.045]"
              >
                <p className="text-sm text-text">{example}</p>
                <span className="mt-3 inline-flex items-center gap-2 text-xs font-medium uppercase text-muted transition group-hover:text-accent">
                  Use brief
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
