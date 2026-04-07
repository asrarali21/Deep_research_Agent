import { forwardRef, type TextareaHTMLAttributes } from "react";

import { cn } from "@/lib/cn";

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea(
  { className, ...props },
  ref,
) {
  return (
    <textarea
      ref={ref}
      className={cn(
        "w-full rounded-field border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-text placeholder:text-muted outline-none transition focus:border-accent/40 focus:bg-white/[0.05]",
        className,
      )}
      {...props}
    />
  );
});
