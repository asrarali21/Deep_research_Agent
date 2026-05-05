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
    <div className="flex h-full flex-col gap-3">
      <SectionCard className="p-4">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <p className="font-display text-lg font-semibold text-text">Research desk</p>
            <p className="mt-1 text-sm leading-6 text-muted">One focused run at a time.</p>
          </div>
          <div className="rounded-panel border border-accent/20 bg-accent/10 p-2 text-accent">
            <Workflow className="h-4 w-4" />
          </div>
        </div>

        <Button className="w-full" onClick={onNewQuery}>
          <Plus className="h-4 w-4" />
          New research
        </Button>
      </SectionCard>

      <SectionCard className="flex-1 p-4">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Layers3 className="h-4 w-4 text-accent" />
            <p className="text-sm font-medium text-text">Current session</p>
          </div>
          <Badge className="capitalize">{status}</Badge>
        </div>
        <p className="line-clamp-8 text-sm leading-6 text-muted">
          {query || "Your next research brief will stay pinned here while the report builds."}
        </p>
        {threadId ? (
          <div className="mt-4 border-t border-white/[0.08] pt-4">
            <p className="mb-1 text-xs font-medium text-muted">Thread</p>
            <p className="break-all font-mono text-xs leading-5 text-muted">{threadId}</p>
          </div>
        ) : null}
      </SectionCard>
    </div>
  );
}
