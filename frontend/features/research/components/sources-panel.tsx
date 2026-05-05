"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ExternalLink, Library } from "lucide-react";

import { SectionCard } from "@/components/layout/section-card";
import { Button } from "@/components/ui/button";
import type { SourceCard } from "@/features/research/types/research";

type SourcesPanelProps = {
  sources: SourceCard[];
  status: string;
  isMobileOpen: boolean;
  onToggleMobile: (open: boolean) => void;
};

function SourcesContent({ sources, status }: { sources: SourceCard[]; status: string }) {
  return (
    <SectionCard className="h-full overflow-hidden">
      <div className="flex items-start justify-between gap-3 border-b border-white/[0.08] p-4">
        <div>
          <p className="text-xs font-medium uppercase text-muted">References</p>
          <p className="mt-1 font-display text-xl font-semibold text-text">
            {sources.length ? `${sources.length} source${sources.length === 1 ? "" : "s"}` : "Sources will appear here"}
          </p>
        </div>
        <div className="rounded-panel border border-accent/20 bg-accent/10 p-2 text-accent">
          <Library className="h-4 w-4" />
        </div>
      </div>

      {sources.length ? (
        <div className="divide-y divide-white/[0.08]">
          {sources.map((source) => (
            <a
              key={source.id}
              href={source.url}
              target="_blank"
              rel="noreferrer noopener"
              className="group block p-4 transition hover:bg-white/[0.045]"
            >
              <div className="mb-2 flex items-center justify-between gap-2">
                <span className="text-xs font-medium uppercase text-muted">Ref {source.index}</span>
                <ExternalLink className="h-4 w-4 text-muted transition group-hover:text-accent" />
              </div>
              <p className="font-medium text-text">{source.title}</p>
              <p className="mt-1 text-sm text-accent">{source.hostname}</p>
              {source.note ? <p className="mt-2 line-clamp-3 text-sm text-muted">{source.note}</p> : null}
            </a>
          ))}
        </div>
      ) : (
        <p className="p-4 text-sm leading-6 text-muted">
          {status === "done" || status === "failed"
            ? "No structured references were detected in the final Markdown report."
            : "References collected during the run will land here."}
        </p>
      )}
    </SectionCard>
  );
}

export function SourcesPanel({ sources, status, isMobileOpen, onToggleMobile }: SourcesPanelProps) {
  return (
    <>
      <div className="hidden xl:block">
        <SourcesContent sources={sources} status={status} />
      </div>

      <div className="xl:hidden">
        <Button variant="secondary" className="mb-3 w-full" onClick={() => onToggleMobile(!isMobileOpen)}>
          <Library className="h-4 w-4" />
          {isMobileOpen ? "Hide sources" : "Show sources"}
        </Button>
        <AnimatePresence initial={false}>
          {isMobileOpen ? (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              transition={{ duration: 0.2 }}
            >
              <SourcesContent sources={sources} status={status} />
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </>
  );
}
