import type { HTMLAttributes } from "react";

export function Tooltip({ title, children, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span title={title} {...props}>
      {children}
    </span>
  );
}
