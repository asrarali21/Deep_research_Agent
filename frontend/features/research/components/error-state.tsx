"use client";

import { TriangleAlert } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { SectionCard } from "@/components/layout/section-card";

type ErrorStateProps = {
  error: string;
  onRetry: () => void;
  onRegenerate: () => void;
  rateLimitResetAt?: number;
  queueRetryAt?: number;
};

function useCountdown(target?: number) {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (!target) {
      return;
    }
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [target]);

  return useMemo(() => {
    if (!target) {
      return null;
    }
    return Math.max(0, Math.ceil((target - now) / 1000));
  }, [now, target]);
}

export function ErrorState({ error, onRetry, onRegenerate, rateLimitResetAt, queueRetryAt }: ErrorStateProps) {
  const rateLimitCountdown = useCountdown(rateLimitResetAt);
  const queueCountdown = useCountdown(queueRetryAt);

  return (
    <SectionCard className="border-danger/25 bg-danger/10 p-5">
      <div className="flex items-start gap-3">
        <div className="inline-flex h-10 w-10 items-center justify-center rounded-panel border border-danger/20 bg-danger/15 text-danger">
          <TriangleAlert className="h-5 w-5" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="font-medium text-text">Research could not complete</p>
          <p className="mt-2 text-sm leading-7 text-muted">{error}</p>
          {rateLimitCountdown !== null ? (
            <p className="mt-2 text-sm text-warning">Rate limit active. Try again in {rateLimitCountdown}s.</p>
          ) : null}
          {queueCountdown !== null ? (
            <p className="mt-2 text-sm text-warning">Queue is full. Retry in {queueCountdown}s.</p>
          ) : null}

          <div className="mt-4 flex flex-wrap gap-2">
            <Button variant="secondary" onClick={onRetry}>
              Retry action
            </Button>
            <Button onClick={onRegenerate}>Start fresh run</Button>
          </div>
        </div>
      </div>
    </SectionCard>
  );
}
