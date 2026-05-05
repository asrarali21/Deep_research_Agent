"use client";

import { Send, Sparkles } from "lucide-react";
import { type KeyboardEvent, useLayoutEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

type InputMode = "query" | "resume";

type InputBoxProps = {
  mode: InputMode;
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
  placeholder: string;
  submitLabel?: string;
};

export function InputBox({
  mode,
  value,
  onChange,
  onSubmit,
  disabled = false,
  placeholder,
  submitLabel,
}: InputBoxProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [isComposing, setIsComposing] = useState(false);

  useLayoutEffect(() => {
    const element = textareaRef.current;
    if (!element) {
      return;
    }

    element.style.height = "0px";
    element.style.height = `${Math.min(element.scrollHeight, 224)}px`;
  }, [value]);

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey || isComposing) {
      return;
    }

    event.preventDefault();
    onSubmit();
  };

  return (
    <div className="rounded-panel border border-white/[0.1] bg-surface-1/95 p-2 shadow-[0_22px_64px_rgba(0,0,0,0.38)]">
      <div className="mb-1 flex items-center gap-2 px-2 pt-1 text-xs font-medium uppercase text-muted">
        {mode === "query" ? <Sparkles className="h-3.5 w-3.5" /> : <Send className="h-3.5 w-3.5" />}
        {mode === "query" ? "Research brief" : "Plan feedback"}
      </div>
      <div className="flex flex-col gap-3">
        <Textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          onCompositionStart={() => setIsComposing(true)}
          onCompositionEnd={() => setIsComposing(false)}
          disabled={disabled}
          placeholder={placeholder}
          className="max-h-56 min-h-[58px] resize-none border-0 bg-transparent px-2 text-base shadow-none focus:bg-transparent sm:text-sm"
        />
        <div className="flex items-center justify-end gap-3 border-t border-white/[0.08] px-1 pt-2">
          <Button onClick={onSubmit} disabled={disabled || !value.trim()}>
            <Send className="h-4 w-4" />
            {submitLabel ?? (mode === "query" ? "Start research" : "Resume research")}
          </Button>
        </div>
      </div>
    </div>
  );
}
