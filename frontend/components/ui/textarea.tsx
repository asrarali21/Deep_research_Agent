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
        "w-full rounded-field border border-white/[0.09] bg-white/[0.035] px-4 py-3 text-sm leading-6 text-text placeholder:text-muted outline-none transition focus:border-accent/45 focus:bg-white/[0.055]",
        className,
      )}
      {...props}
    />
  );
});
