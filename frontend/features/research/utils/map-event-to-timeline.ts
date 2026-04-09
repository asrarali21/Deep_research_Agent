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
      return makeTimelineItem(
        "plan",
        "Research plan created",
        event.data.required_sections?.length ? "Quality gate sections were also defined for the final report." : undefined,
        [...event.data.plan, ...(event.data.required_sections ?? []).map((section) => `Section: ${section}`)],
      );
    case "agent":
      return makeTimelineItem(
        "research",
        event.data.task ? `Completed research task: ${event.data.task}` : "Sub-research completed",
        event.data.evidence_count || event.data.source_count
          ? `Captured ${event.data.evidence_count ?? 0} evidence cards across ${event.data.source_count ?? 0} scraped sources.`
          : undefined,
      );
    case "source_batch":
      return makeTimelineItem(
        "research",
        event.data.task ? `Collected sources for: ${event.data.task}` : "Collected live sources",
        `Observed ${event.data.discovered_count} candidate URLs and ${event.data.scraped_count} scraped sources.`,
        [...event.data.scraped_sources, ...event.data.discovered_sources].slice(0, 8),
      );
    case "evidence_batch":
      return makeTimelineItem(
        "research",
        event.data.task ? `Extracted evidence for: ${event.data.task}` : "Extracted evidence",
        `Captured ${event.data.evidence_count} evidence cards and ${event.data.finding_count} findings.`,
        event.data.coverage_tags,
      );
    case "evaluate":
      return makeTimelineItem(
        "evaluate",
        event.data.gap_count ? "Gap analysis found follow-ups" : "Gap analysis found enough evidence",
        event.data.quality_summary,
        event.data.gaps,
      );
    case "outline":
      return makeTimelineItem("plan", "Report outline created", undefined, event.data.sections);
    case "section_draft":
      return makeTimelineItem(
        "synthesize",
        `Drafted section: ${event.data.section}`,
        `${event.data.char_count} characters generated for this section.`,
      );
    case "synthesize":
      return makeTimelineItem(
        "synthesize",
        event.data.status === "final_editing_report" ? "Final editorial pass in progress" : "Synthesizing final report",
      );
    case "failed":
      return makeTimelineItem("error", "Research failed", event.data.error);
    default:
      return null;
  }
}
