import { z } from "zod";

const statusSchema = z.enum(["queued", "running", "paused", "done", "failed"]);

export const researchStatusSchema = z.object({
  thread_id: z.string(),
  status: statusSchema,
  queue_position: z.number().nullable(),
  retry_count: z.number(),
  provider_switch_count: z.number(),
  current_plan: z.array(z.string()),
  extracted_facts_count: z.number(),
  last_error: z.string(),
});

export const researchEventSchema = z.discriminatedUnion("event", [
  z.object({
    event: z.literal("queued"),
    data: z.object({ thread_id: z.string(), status: z.literal("queued"), resume: z.literal(true).optional() }),
  }),
  z.object({
    event: z.literal("started"),
    data: z.object({ thread_id: z.string(), status: z.literal("running"), worker: z.number() }),
  }),
  z.object({
    event: z.literal("plan"),
    data: z.object({ thread_id: z.string(), task_count: z.number(), plan: z.array(z.string()) }),
  }),
  z.object({
    event: z.literal("agent"),
    data: z.object({ thread_id: z.string(), status: z.literal("sub_agent_complete") }),
  }),
  z.object({
    event: z.literal("evaluate"),
    data: z.object({ thread_id: z.string(), gaps: z.array(z.string()), gap_count: z.number() }),
  }),
  z.object({
    event: z.literal("synthesize"),
    data: z.object({ thread_id: z.string(), status: z.literal("finalizing_report") }),
  }),
  z.object({
    event: z.literal("report"),
    data: z.object({ thread_id: z.string(), report: z.string() }),
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
