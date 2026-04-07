import type { ResearchEvent, TimelineItem } from "@/features/research/types/research";

function makeTimelineItem(
  kind: TimelineItem["kind"],
  title: string,
  description?: string,
  meta?: string[],
): TimelineItem {
  return {
    id: `${kind}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    kind,
    title,
    description,
    meta,
    createdAt: Date.now(),
  };
}

export function mapEventToTimeline(event: ResearchEvent): TimelineItem | null {
  switch (event.event) {
    case "queued":
      return makeTimelineItem("queued", event.data.resume ? "Re-queued after feedback" : "Queued for research");
    case "started":
      return makeTimelineItem("system", "Research worker started", `Worker ${event.data.worker} claimed the job.`);
    case "plan":
      return makeTimelineItem("plan", "Research plan created", undefined, event.data.plan);
    case "agent":
      return makeTimelineItem("research", "Sub-research completed");
    case "evaluate":
      return makeTimelineItem(
        "evaluate",
        event.data.gap_count ? "Gap analysis found follow-ups" : "Gap analysis found enough evidence",
        undefined,
        event.data.gaps,
      );
    case "synthesize":
      return makeTimelineItem("synthesize", "Synthesizing final report");
    case "failed":
      return makeTimelineItem("error", "Research failed", event.data.error);
    default:
      return null;
  }
}
