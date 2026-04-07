"use client";

import { ArrowDown } from "lucide-react";

import { IconButton } from "@/components/ui/icon-button";

type ScrollToBottomProps = {
  visible: boolean;
  onClick: () => void;
};

export function ScrollToBottom({ visible, onClick }: ScrollToBottomProps) {
  if (!visible) {
    return null;
  }

  return (
    <div className="pointer-events-none sticky bottom-6 flex justify-center">
      <IconButton onClick={onClick} className="pointer-events-auto bg-surface-2/90 text-text shadow-panel">
        <ArrowDown className="h-4 w-4" />
      </IconButton>
    </div>
  );
}
