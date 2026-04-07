import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ThinkingSection } from "@/features/research/components/thinking-section";

describe("ThinkingSection", () => {
  it("renders timeline items and toggles via header button", () => {
    const onToggle = vi.fn();

    render(
      <ThinkingSection
        items={[
          {
            id: "1",
            kind: "plan",
            title: "Research plan created",
            meta: ["Task A"],
            createdAt: Date.now(),
          },
        ]}
        status="running"
        open
        onToggle={onToggle}
      />,
    );

    expect(screen.getByText("Research plan created")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /research process/i }));
    expect(onToggle).toHaveBeenCalledTimes(1);
  });
});
