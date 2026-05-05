import { forwardRef, type HTMLAttributes } from "react";

import { cn } from "@/lib/cn";

export const SectionCard = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(function SectionCard(
  { className, ...props },
  ref,
) {
  return (
    <div
      ref={ref}
      className={cn(
        "glass-panel rounded-panel border border-white/[0.09] shadow-panel ring-1 ring-black/20",
        className,
      )}
      {...props}
    />
  );
});
