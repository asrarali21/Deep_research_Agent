import { forwardRef, type HTMLAttributes } from "react";

import { cn } from "@/lib/cn";

export const ScrollArea = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(function ScrollArea(
  { className, ...props },
  ref,
) {
  return <div ref={ref} className={cn("overflow-y-auto", className)} {...props} />;
});
