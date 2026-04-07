import { describe, expect, it } from "vitest";

import { parseSourcesFromMarkdown } from "@/features/research/utils/parse-sources";

describe("parseSourcesFromMarkdown", () => {
  it("prefers numbered references when present", () => {
    const markdown = `
# Report

Content.

1. [OpenAI](https://openai.com/research)
2. Bloomberg https://www.bloomberg.com/news/articles/example
`;

    const sources = parseSourcesFromMarkdown(markdown);

    expect(sources).toHaveLength(2);
    expect(sources[0]).toMatchObject({
      index: 1,
      hostname: "openai.com",
      url: "https://openai.com/research",
    });
  });
});
