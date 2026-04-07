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
});
