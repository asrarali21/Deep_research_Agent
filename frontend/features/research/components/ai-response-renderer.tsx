"use client";

import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

import { SectionCard } from "@/components/layout/section-card";
import { Skeleton } from "@/components/ui/skeleton";
import { markdownComponents } from "@/features/research/utils/markdown-components";
import { fadeUp } from "@/lib/motion";

type AIResponseRendererProps = {
  markdown: string;
  isStreamingReveal: boolean;
};

export function AIResponseRenderer({ markdown, isStreamingReveal }: AIResponseRendererProps) {
  if (!markdown) {
    return (
      <SectionCard className="p-6">
        <div className="space-y-3">
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-[92%]" />
          <Skeleton className="h-4 w-[76%]" />
        </div>
      </SectionCard>
    );
  }

  return (
    <motion.div {...fadeUp}>
      <SectionCard className="overflow-hidden p-6 sm:p-8">
        <div className="mb-5 flex items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-muted">Final report</p>
            <p className="mt-1 font-display text-2xl font-semibold tracking-tight text-text">Structured answer</p>
          </div>
          {isStreamingReveal ? (
            <span className="rounded-full border border-accent/20 bg-accent/10 px-3 py-1 text-xs text-accent">
              Revealing report
            </span>
          ) : null}
        </div>

        <div className="report-prose">
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]} components={markdownComponents}>
            {markdown}
          </ReactMarkdown>
        </div>
      </SectionCard>
    </motion.div>
  );
}
