"use client";

import { Check, Copy, RefreshCw, Square, TriangleAlert } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { copyToClipboard } from "@/features/research/utils/copy";

type ToolbarProps = {
  canStop: boolean;
  canRetry: boolean;
  canRegenerate: boolean;
  report: string;
  onStop: () => void;
  onRetry: () => void;
  onRegenerate: () => void;
};

export function Toolbar({ canStop, canRetry, canRegenerate, report, onStop, onRetry, onRegenerate }: ToolbarProps) {
  const [copied, setCopied] = useState(false);

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        variant="secondary"
        size="sm"
        onClick={async () => {
          await copyToClipboard(report);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1200);
        }}
        disabled={!report}
      >
        {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
        {copied ? "Copied" : "Copy report"}
      </Button>
      <Button variant="secondary" size="sm" onClick={onRegenerate} disabled={!canRegenerate}>
        <RefreshCw className="h-4 w-4" />
        Regenerate
      </Button>
      <Button variant="secondary" size="sm" onClick={onRetry} disabled={!canRetry}>
        <TriangleAlert className="h-4 w-4" />
        Retry
      </Button>
      <Button variant="danger" size="sm" onClick={onStop} disabled={!canStop}>
        <Square className="h-4 w-4" />
        Stop
      </Button>
    </div>
  );
}
