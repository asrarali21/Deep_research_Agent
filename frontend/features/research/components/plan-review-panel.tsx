"use client";

import { Plus, Wand2, X } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { Textarea } from "@/components/ui/textarea";

type PlanReviewPanelProps = {
  plan: string[];
  onChange: (plan: string[]) => void;
  onResume: (note: string) => void;
  onContinue: () => void;
  disabled?: boolean;
};

export function PlanReviewPanel({ plan, onChange, onResume, onContinue, disabled = false }: PlanReviewPanelProps) {
  const [items, setItems] = useState(plan);
  const [note, setNote] = useState("");

  useEffect(() => {
    setItems(plan);
  }, [plan]);

  const updateItems = (next: string[]) => {
    setItems(next);
    onChange(next);
  };

  return (
    <div className="rounded-panel border border-warning/25 bg-warning/10 p-5 shadow-panel">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <p className="font-display text-lg font-semibold text-text">Approve or revise the plan</p>
          <p className="mt-1 text-sm text-muted">Adjust the tasks before the run continues.</p>
        </div>
      </div>

      <div className="space-y-3">
        {items.map((item, index) => (
          <div key={`${index}-${item}`} className="flex gap-3">
            <div className="mt-3 text-xs text-muted">{index + 1}.</div>
            <Textarea
              value={item}
              onChange={(event) => {
                const next = [...items];
                next[index] = event.target.value;
                updateItems(next);
              }}
              className="min-h-[68px]"
            />
            <IconButton
              aria-label={`Remove task ${index + 1}`}
              className="self-start rounded-panel"
              onClick={() => updateItems(items.filter((_, itemIndex) => itemIndex !== index))}
            >
              <X className="h-4 w-4" />
            </IconButton>
          </div>
        ))}

        <Button variant="secondary" onClick={() => updateItems([...items, ""])}>
          <Plus className="h-4 w-4" />
          Add task
        </Button>

        <Textarea
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder="Add guidance for the next pass."
          className="min-h-[110px]"
        />

        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={onContinue} disabled={disabled}>
            Continue as-is
          </Button>
          <Button onClick={() => onResume(note)} disabled={disabled}>
            <Wand2 className="h-4 w-4" />
            Revise and continue
          </Button>
        </div>
      </div>
    </div>
  );
}
