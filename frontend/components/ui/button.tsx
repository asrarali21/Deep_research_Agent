import { forwardRef, type ButtonHTMLAttributes } from "react";

import { cn } from "@/lib/cn";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md" | "lg";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
};

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-accent text-[#10130f] shadow-[0_10px_28px_rgba(99,222,169,0.18)] hover:bg-accent-2 focus-visible:ring-accent/50",
  secondary:
    "border border-white/[0.09] bg-white/[0.045] text-text hover:border-white/[0.16] hover:bg-white/[0.08] focus-visible:ring-white/20",
  ghost:
    "bg-transparent text-muted hover:bg-white/[0.055] hover:text-text focus-visible:ring-white/20",
  danger:
    "bg-danger/85 text-white shadow-[0_10px_28px_rgba(248,113,113,0.14)] hover:bg-danger focus-visible:ring-danger/40",
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: "h-9 px-3 text-sm",
  md: "h-11 px-4 text-sm",
  lg: "h-12 px-5 text-base",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant = "primary", size = "md", type = "button", ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-panel font-medium transition duration-200 focus-visible:outline-none focus-visible:ring-2 disabled:cursor-not-allowed disabled:opacity-50",
        variantClasses[variant],
        sizeClasses[size],
        className,
      )}
      {...props}
    />
  );
});
