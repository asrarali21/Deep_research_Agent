import { Layers3, Plus, Workflow } from "lucide-react";

import { SectionCard } from "@/components/layout/section-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

type SidebarProps = {
  query: string;
  status: string;
  threadId?: string;
  onNewQuery: () => void;
};

export function Sidebar({ query, status, threadId, onNewQuery }: SidebarProps) {
  return (
    <div className="flex h-full flex-col gap-4">
      <SectionCard className="p-5">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <p className="font-display text-lg font-semibold tracking-tight text-text">Research workspace</p>
            <p className="text-sm text-muted">Single-session mode for the current backend capabilities.</p>
          </div>
          <Workflow className="h-5 w-5 text-accent" />
        </div>

        <Button className="w-full" onClick={onNewQuery}>
          <Plus className="h-4 w-4" />
          New research
        </Button>
      </SectionCard>

      <SectionCard className="p-5">
        <div className="mb-3 flex items-center gap-2">
          <Layers3 className="h-4 w-4 text-accent" />
          <p className="text-sm font-medium text-text">Current session</p>
        </div>
        <Badge className="mb-3 capitalize">{status}</Badge>
        <p className="line-clamp-6 text-sm text-muted">{query || "No research brief yet."}</p>
        {threadId ? <p className="mt-4 break-all font-mono text-xs text-muted">{threadId}</p> : null}
      </SectionCard>

      <SectionCard className="p-5">
        <p className="text-sm font-medium text-text">History later</p>
        <p className="mt-2 text-sm leading-7 text-muted">
          The backend does not expose history or report retrieval yet, so this rail stays focused on the active run.
        </p>
      </SectionCard>
    </div>
  );
}
