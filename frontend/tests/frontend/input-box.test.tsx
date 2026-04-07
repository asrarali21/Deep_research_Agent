import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { InputBox } from "@/features/research/components/input-box";

describe("InputBox", () => {
  it("submits on Enter without Shift", () => {
    const onSubmit = vi.fn();

    render(
      <InputBox
        mode="query"
        value="Research this company"
        onChange={() => undefined}
        onSubmit={onSubmit}
        placeholder="placeholder"
      />,
    );

    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });

    expect(onSubmit).toHaveBeenCalledTimes(1);
  });
});
