"use client";

import { Plus, Wand2 } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
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
    <div className="rounded-panel border border-warning/20 bg-warning/10 p-5">
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <p className="font-display text-lg font-semibold text-text">Approve or revise the plan</p>
          <p className="mt-1 text-sm text-muted">
            The backend is paused at the planning checkpoint. Edit tasks if needed, then resume.
          </p>
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
            <Button
              variant="ghost"
              className="self-start"
              onClick={() => updateItems(items.filter((_, itemIndex) => itemIndex !== index))}
            >
              Remove
            </Button>
          </div>
        ))}

        <Button variant="secondary" onClick={() => updateItems([...items, ""])}>
          <Plus className="h-4 w-4" />
          Add task
        </Button>

        <Textarea
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder="Optional guidance for the orchestrator. Example: prioritize company filings over news articles."
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
