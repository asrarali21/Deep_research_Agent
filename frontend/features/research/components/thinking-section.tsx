"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Bot, CheckCircle2, CircleDashed, FileSearch, Sparkles, TriangleAlert } from "lucide-react";

import { Collapsible } from "@/components/ui/collapsible";
import type { TimelineItem } from "@/features/research/types/research";
import { fadeUp, staggerParent } from "@/lib/motion";

type ThinkingSectionProps = {
  items: TimelineItem[];
  status: string;
  open: boolean;
  onToggle: () => void;
};

function iconForKind(kind: TimelineItem["kind"]) {
  switch (kind) {
    case "queued":
      return CircleDashed;
    case "plan":
      return Sparkles;
    case "research":
      return FileSearch;
    case "evaluate":
      return CheckCircle2;
    case "synthesize":
      return Bot;
    case "error":
      return TriangleAlert;
    default:
      return CircleDashed;
  }
}

export function ThinkingSection({ items, status, open, onToggle }: ThinkingSectionProps) {
  return (
    <Collapsible
      title="Research process"
      subtitle={status === "done" ? "Completed timeline" : "Live workflow timeline"}
      open={open}
      onToggle={onToggle}
    >
      {items.length ? (
        <motion.div className="space-y-3" variants={staggerParent} initial="initial" animate="animate">
          <AnimatePresence initial={false}>
            {items.map((item) => {
              const Icon = iconForKind(item.kind);
              return (
                <motion.div
                  key={item.id}
                  {...fadeUp}
                  className="rounded-[18px] border border-white/10 bg-white/[0.03] p-4"
                >
                  <div className="flex gap-3">
                    <div className="mt-0.5 inline-flex h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] text-accent">
                      <Icon className="h-4 w-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="font-medium text-text">{item.title}</p>
                      {item.description ? <p className="mt-1 text-sm text-muted">{item.description}</p> : null}
                      {item.meta?.length ? (
                        <div className="mt-3 flex flex-wrap gap-2">
                          {item.meta.map((entry) => (
                            <span
                              key={`${item.id}-${entry}`}
                              className="rounded-full border border-white/10 bg-black/20 px-2.5 py-1 text-xs text-muted"
                            >
                              {entry}
                            </span>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </motion.div>
      ) : (
        <p className="text-sm text-muted">Start a research run to see the orchestration timeline appear here.</p>
      )}
    </Collapsible>
  );
}
