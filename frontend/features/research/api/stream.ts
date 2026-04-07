import { researchEventSchema } from "@/features/research/api/schemas";
import type { ResearchEvent } from "@/features/research/types/research";

type ParseResult = {
  events: ResearchEvent[];
  rest: string;
};

export function extractSseEvents(buffer: string): ParseResult {
  const frames = buffer.split("\n\n");
  const rest = frames.pop() ?? "";
  const events = frames.flatMap((frame) => {
    const lines = frame.split("\n");
    const eventName = lines.find((line) => line.startsWith("event:"))?.replace("event:", "").trim();
    const dataLines = lines
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.replace("data:", "").trim());

    if (!eventName || !dataLines.length) {
      return [];
    }

    try {
      const payload = JSON.parse(dataLines.join("\n"));
      const parsed = researchEventSchema.parse({ event: eventName, data: payload });
      return [parsed];
    } catch {
      return [];
    }
  });

  return { events, rest };
}

export async function postSse<TBody>(
  url: string,
  body: TBody,
  opts: {
    signal?: AbortSignal;
    onHeaders?: (headers: Headers) => void;
    onEvent: (event: ResearchEvent) => void;
  },
) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal: opts.signal,
  });

  if (!response.ok || !response.body) {
    throw response;
  }

  opts.onHeaders?.(response.headers);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const { events, rest } = extractSseEvents(buffer);
    buffer = rest;

    for (const event of events) {
      opts.onEvent(event);
    }
  }
}
