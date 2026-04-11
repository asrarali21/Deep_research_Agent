import { describe, expect, it } from "vitest";

import { mapEventToTimeline } from "@/features/research/utils/map-event-to-timeline";

describe("mapEventToTimeline", () => {
  it("maps plan events into timeline entries with metadata", () => {
    const item = mapEventToTimeline({
      event: "plan",
      data: { thread_id: "abc", task_count: 2, plan: ["Investigate pricing", "Investigate market size"] },
    });

    expect(item).toMatchObject({
      kind: "plan",
      title: "Research plan created",
      meta: ["Investigate pricing", "Investigate market size"],
    });
  });

  it("maps live source events into research timeline entries", () => {
    const item = mapEventToTimeline({
      event: "source_batch",
      data: {
        thread_id: "abc",
        task: "Diet evidence",
        discovered_sources: ["https://example.com/a"],
        scraped_sources: ["https://example.com/b"],
        discovered_count: 1,
        scraped_count: 1,
      },
    });

    expect(item).toMatchObject({
      kind: "research",
      title: "Collected sources for: Diet evidence",
      meta: ["https://example.com/b", "https://example.com/a"],
    });
  });

  it("maps quota wait events into system timeline entries", () => {
    const item = mapEventToTimeline({
      event: "waiting_for_quota",
      data: {
        thread_id: "abc",
        status: "waiting_for_quota",
        task_type: "planner",
        retry_after_seconds: 120,
        available_at: Date.now() + 120_000,
        error: "All providers are temporarily unavailable",
      },
    });

    expect(item).toMatchObject({
      kind: "system",
      title: "Waiting for provider quota reset",
    });
  });
});
