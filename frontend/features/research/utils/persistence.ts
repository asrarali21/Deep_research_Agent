import type { ResearchSessionState, SessionSnapshot } from "@/features/research/types/research";

const STORAGE_KEY = "deep-research-session-v1";

export function saveSessionSnapshot(state: ResearchSessionState) {
  if (typeof window === "undefined") {
    return;
  }

  const snapshot: SessionSnapshot = {
    ...state,
    status: state.status,
  };

  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
}

export function loadSessionSnapshot(): SessionSnapshot | null {
  if (typeof window === "undefined") {
    return null;
  }

  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return null;
  }

  try {
    return JSON.parse(raw) as SessionSnapshot;
  } catch {
    return null;
  }
}

export function clearSessionSnapshot() {
  if (typeof window === "undefined") {
    return;
  }

  window.localStorage.removeItem(STORAGE_KEY);
}
