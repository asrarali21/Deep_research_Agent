import { describe, expect, it } from "vitest";

import { extractSseEvents } from "@/features/research/api/stream";

describe("extractSseEvents", () => {
  it("parses complete SSE frames and preserves trailing partial data", () => {
    const input = [
      'event: queued\ndata: {"thread_id":"abc","status":"queued"}\n\n',
      'event: plan\ndata: {"thread_id":"abc","task_count":2,"plan":["A","B"]}\n\n',
      'event: synthesize\ndata: {"thread_id":"abc","status":"finalizing_report"}\n',
    ].join("");

    const result = extractSseEvents(input);

    expect(result.events).toHaveLength(2);
    expect(result.events[0]).toMatchObject({ event: "queued", data: { thread_id: "abc" } });
    expect(result.events[1]).toMatchObject({ event: "plan", data: { plan: ["A", "B"] } });
    expect(result.rest).toContain("event: synthesize");
  });
});
