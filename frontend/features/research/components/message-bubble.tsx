import type { PropsWithChildren } from "react";

import { cn } from "@/lib/cn";

type MessageBubbleProps = PropsWithChildren<{
  variant: "user" | "system";
  title?: string;
  className?: string;
}>;

export function MessageBubble({ variant, title, className, children }: MessageBubbleProps) {
  return (
    <div
      className={cn(
        "rounded-panel border px-5 py-4",
        variant === "user"
          ? "border-accent/20 bg-accent/10"
          : "border-white/10 bg-white/[0.03]",
        className,
      )}
    >
      {title ? <p className="mb-2 text-xs uppercase tracking-[0.22em] text-muted">{title}</p> : null}
      <div className="text-sm text-text">{children}</div>
    </div>
  );
}
