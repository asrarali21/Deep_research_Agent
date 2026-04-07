"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { type PropsWithChildren } from "react";

import { cn } from "@/lib/cn";

type CollapsibleProps = PropsWithChildren<{
  title: string;
  open: boolean;
  onToggle: () => void;
  className?: string;
  subtitle?: string;
}>;

export function Collapsible({ title, subtitle, open, onToggle, className, children }: CollapsibleProps) {
  return (
    <div className={cn("rounded-panel border border-white/10 bg-white/[0.02]", className)}>
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left"
      >
        <div>
          <p className="font-medium text-text">{title}</p>
          {subtitle ? <p className="text-sm text-muted">{subtitle}</p> : null}
        </div>
        <ChevronDown className={cn("h-4 w-4 text-muted transition", open && "rotate-180")} />
      </button>

      <AnimatePresence initial={false}>
        {open ? (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4">{children}</div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
