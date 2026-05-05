import { Activity, DatabaseZap, Route, TimerReset } from "lucide-react";

import { Badge } from "@/components/ui/badge";

type StatusStripProps = {
  status: string;
  queuePosition?: number | null;
  retryCount: number;
  providerSwitchCount: number;
  factsCount: number;
  evidenceCount: number;
};

function statusVariant(status: string): "accent" | "success" | "warning" | "danger" | "neutral" {
  switch (status) {
    case "running":
      return "accent";
    case "done":
      return "success";
    case "waiting_for_quota":
    case "paused":
    case "queued":
      return "warning";
    case "failed":
      return "danger";
    default:
      return "neutral";
  }
}

const metricClass = "inline-flex items-center gap-2 rounded-panel border border-white/[0.08] bg-white/[0.035] px-3 py-2 text-sm text-muted";

export function StatusStrip({
  status,
  queuePosition,
  retryCount,
  providerSwitchCount,
  factsCount,
  evidenceCount,
}: StatusStripProps) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-panel border border-white/[0.08] bg-surface-1/70 p-2 shadow-panel">
      <Badge variant={statusVariant(status)} className="capitalize">
        {status}
      </Badge>
      {typeof queuePosition === "number" && queuePosition >= 0 ? (
        <div className={metricClass}>
          <Activity className="h-4 w-4" />
          Queue position {queuePosition}
        </div>
      ) : null}
      <div className={metricClass}>
        <TimerReset className="h-4 w-4" />
        Retries {retryCount}
      </div>
      <div className={metricClass}>
        <Route className="h-4 w-4" />
        Provider switches {providerSwitchCount}
      </div>
      <div className={metricClass}>
        <DatabaseZap className="h-4 w-4" />
        Facts extracted {factsCount}
      </div>
      <div className={metricClass}>
        <DatabaseZap className="h-4 w-4" />
        Evidence cards {evidenceCount}
      </div>
    </div>
  );
}
