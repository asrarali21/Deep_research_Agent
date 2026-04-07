import { researchStatusSchema } from "@/features/research/api/schemas";
import { postSse } from "@/features/research/api/stream";
import type { ResearchEvent, ResearchStatus } from "@/features/research/types/research";
import { getApiBaseUrl } from "@/lib/env";

export type ApiError = Error & {
  status?: number;
  retryAfterSeconds?: number;
};

function parseRetryAfter(response: Response) {
  const raw = response.headers.get("Retry-After");
  if (!raw) {
    return undefined;
  }
  const numeric = Number(raw);
  return Number.isFinite(numeric) ? numeric : undefined;
}

async function toApiError(response: Response): Promise<ApiError> {
  let message = `Request failed with status ${response.status}`;
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload?.detail) {
      message = payload.detail;
    }
  } catch {
    // Ignore JSON parse errors and keep the generic message.
  }

  const error = new Error(message) as ApiError;
  error.status = response.status;
  error.retryAfterSeconds = parseRetryAfter(response);
  return error;
}

async function fetchStatus(threadId: string): Promise<ResearchStatus> {
  const response = await fetch(`${getApiBaseUrl()}/api/research/status/${threadId}`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    throw await toApiError(response);
  }

  const payload = await response.json();
  return researchStatusSchema.parse(payload);
}

async function streamRequest<TBody>(
  path: string,
  body: TBody,
  onEvent: (event: ResearchEvent) => void,
  signal?: AbortSignal,
) {
  try {
    await postSse(`${getApiBaseUrl()}${path}`, body, {
      signal,
      onEvent,
    });
  } catch (error) {
    if (error instanceof Response) {
      throw await toApiError(error);
    }
    throw error;
  }
}

export const researchApi = {
  fetchStatus,
  startStream(query: string, onEvent: (event: ResearchEvent) => void, signal?: AbortSignal) {
    return streamRequest("/api/research/start", { query }, onEvent, signal);
  },
  resumeStream(threadId: string, feedback: string, onEvent: (event: ResearchEvent) => void, signal?: AbortSignal) {
    return streamRequest("/api/research/resume", { thread_id: threadId, feedback }, onEvent, signal);
  },
};
