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
        "max-w-[78ch] rounded-panel border px-5 py-4 shadow-panel",
        variant === "user"
          ? "ml-auto border-accent/25 bg-accent/10"
          : "border-white/[0.09] bg-surface-1/72",
        className,
      )}
    >
      {title ? <p className="mb-2 text-xs font-medium uppercase text-muted">{title}</p> : null}
      <div className="text-sm leading-7 text-text">{children}</div>
    </div>
  );
}
