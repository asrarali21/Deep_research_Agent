import { z } from "zod";

const statusSchema = z.enum(["queued", "running", "waiting_for_quota", "paused", "done", "failed"]);

export const researchStatusSchema = z.object({
  thread_id: z.string(),
  status: statusSchema,
  queue_position: z.number().nullable(),
  retry_count: z.number(),
  provider_switch_count: z.number(),
  current_plan: z.array(z.string()),
  required_sections: z.array(z.string()).optional(),
  extracted_facts_count: z.number(),
  evidence_card_count: z.number().optional(),
  quota_wait_until: z.number().nullable().optional(),
  quota_retry_after_seconds: z.number().optional(),
  waiting_task_type: z.string().optional(),
  last_error: z.string(),
});

export const researchEventSchema = z.discriminatedUnion("event", [
  z.object({
    event: z.literal("queued"),
    data: z.object({
      thread_id: z.string(),
      status: z.literal("queued"),
      resume: z.literal(true).optional(),
      reason: z.string().optional(),
    }),
  }),
  z.object({
    event: z.literal("started"),
    data: z.object({ thread_id: z.string(), status: z.literal("running"), worker: z.number() }),
  }),
  z.object({
    event: z.literal("waiting_for_quota"),
    data: z.object({
      thread_id: z.string(),
      status: z.literal("waiting_for_quota"),
      task_type: z.string(),
      retry_after_seconds: z.number(),
      available_at: z.number(),
      error: z.string(),
    }),
  }),
  z.object({
    event: z.literal("plan"),
    data: z.object({
      thread_id: z.string(),
      task_count: z.number(),
      plan: z.array(z.string()),
      required_sections: z.array(z.string()).optional(),
    }),
  }),
  z.object({
    event: z.literal("agent"),
    data: z.object({
      thread_id: z.string(),
      status: z.literal("sub_agent_complete"),
      task: z.string().optional(),
      evidence_count: z.number().optional(),
      source_count: z.number().optional(),
    }),
  }),
  z.object({
    event: z.literal("source_batch"),
    data: z.object({
      thread_id: z.string(),
      task: z.string(),
      discovered_sources: z.array(z.string()),
      scraped_sources: z.array(z.string()),
      discovered_count: z.number(),
      scraped_count: z.number(),
    }),
  }),
  z.object({
    event: z.literal("evidence_batch"),
    data: z.object({
      thread_id: z.string(),
      task: z.string(),
      evidence_count: z.number(),
      finding_count: z.number(),
      coverage_tags: z.array(z.string()),
    }),
  }),
  z.object({
    event: z.literal("evaluate"),
    data: z.object({
      thread_id: z.string(),
      gaps: z.array(z.string()),
      gap_count: z.number(),
      quality_summary: z.string().optional(),
    }),
  }),
  z.object({
    event: z.literal("outline"),
    data: z.object({ thread_id: z.string(), sections: z.array(z.string()) }),
  }),
  z.object({
    event: z.literal("section_draft"),
    data: z.object({
      thread_id: z.string(),
      section: z.string(),
      status: z.literal("section_drafted"),
      char_count: z.number(),
    }),
  }),
  z.object({
    event: z.literal("evidence_brief"),
    data: z.object({
      thread_id: z.string(),
      section_count: z.number(),
      priority_sections: z.array(z.string()),
    }),
  }),
  z.object({
    event: z.literal("section_verification"),
    data: z.object({
      thread_id: z.string(),
      verified_sections: z.array(z.string()),
      verification_count: z.number(),
    }),
  }),
  z.object({
    event: z.literal("synthesize"),
    data: z.object({
      thread_id: z.string(),
      status: z.union([z.literal("finalizing_report"), z.literal("final_editing_report")]),
    }),
  }),
  z.object({
    event: z.literal("report"),
    data: z.object({
      thread_id: z.string(),
      report: z.string(),
      structured_references: z
        .array(
          z.object({
            id: z.string(),
            evidence_ids: z.array(z.string()).optional(),
            title: z.string(),
            url: z.string(),
            hostname: z.string().optional(),
            source_type: z.string().optional(),
            verification_status: z.string().optional(),
          }),
        )
        .optional(),
    }),
  }),
  z.object({
    event: z.literal("paused"),
    data: z.object({ thread_id: z.string(), status: z.literal("paused") }),
  }),
  z.object({
    event: z.literal("done"),
    data: z.object({ thread_id: z.string(), status: z.literal("done") }),
  }),
  z.object({
    event: z.literal("failed"),
    data: z.object({ thread_id: z.string(), status: z.literal("failed"), error: z.string() }),
  }),
  z.object({
    event: z.literal("heartbeat"),
    data: researchStatusSchema,
  }),
]);
