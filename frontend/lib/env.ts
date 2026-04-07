const fallbackApiBaseUrl = "http://localhost:8000";

export function getApiBaseUrl() {
  const value = process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || fallbackApiBaseUrl;
  return value.endsWith("/") ? value.slice(0, -1) : value;
}
