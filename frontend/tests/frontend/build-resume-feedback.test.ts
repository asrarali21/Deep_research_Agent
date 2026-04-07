import { describe, expect, it } from "vitest";

import { buildResumeFeedback } from "@/features/research/utils/build-resume-feedback";

describe("buildResumeFeedback", () => {
  it("returns an empty string when nothing changed", () => {
    expect(buildResumeFeedback(["A", "B"], ["A", "B"], "")).toBe("");
  });

  it("builds a revision prompt when tasks change", () => {
    const feedback = buildResumeFeedback(["A", "B"], ["A", "C"], "Prioritize primary sources");

    expect(feedback).toContain("- Remove: B");
    expect(feedback).toContain("- Add: C");
    expect(feedback).toContain("User note: Prioritize primary sources");
  });
});
