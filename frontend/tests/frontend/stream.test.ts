import { describe, expect, it } from "vitest";

import { extractSseEvents } from "@/features/research/api/stream";

describe("extractSseEvents", () => {
  it("parses complete SSE frames and preserves trailing partial data", () => {
    const input = [
      'event: queued\ndata: {"thread_id":"abc","status":"queued"}\n\n',
      'event: plan\ndata: {"thread_id":"abc","task_count":2,"plan":["A","B"]}\n\n',
      'event: source_batch\ndata: {"thread_id":"abc","task":"Task A","discovered_sources":["https://example.com"],"scraped_sources":[],"discovered_count":1,"scraped_count":0}\n\n',
      'event: synthesize\ndata: {"thread_id":"abc","status":"finalizing_report"}\n',
    ].join("");

    const result = extractSseEvents(input);

    expect(result.events).toHaveLength(3);
    expect(result.events[0]).toMatchObject({ event: "queued", data: { thread_id: "abc" } });
    expect(result.events[1]).toMatchObject({ event: "plan", data: { plan: ["A", "B"] } });
    expect(result.events[2]).toMatchObject({ event: "source_batch", data: { discovered_count: 1 } });
    expect(result.rest).toContain("event: synthesize");
  });
});
