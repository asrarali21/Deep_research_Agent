import type { SourceCard } from "@/features/research/types/research";

function sanitizeUrl(url: string) {
  const trimmed = url.trim().replace(/[)>.,]+$/, "");
  return trimmed;
}

function tryCreateSource(line: string, index: number): SourceCard | null {
  const match = line.match(/\((https?:\/\/[^)]+)\)|https?:\/\/\S+/);
  const url = sanitizeUrl(match?.[1] ?? match?.[0] ?? "");
  if (!url) {
    return null;
  }

  try {
    const hostname = new URL(url).hostname.replace(/^www\./, "");
    const title = line
      .replace(/^\d+\.\s*/, "")
      .replace(/\[[^\]]*\]\((https?:\/\/[^)]+)\)/g, (full) => full.replace(/\((https?:\/\/[^)]+)\)/, "").trim())
      .replace(url, "")
      .replace(/[\[\]()]/g, "")
      .trim();

    return {
      id: `${index}-${url}`,
      index: index + 1,
      title: title || hostname,
      url,
      hostname,
      note: line.trim(),
    };
  } catch {
    return null;
  }
}

export function parseSourcesFromMarkdown(markdown: string): SourceCard[] {
  const lines = markdown.split("\n");
  const numberedReferences = lines.filter((line) => /^\d+\.\s+/.test(line.trim()));
  const candidates = numberedReferences.length
    ? numberedReferences
    : lines.filter((line) => /https?:\/\/\S+/.test(line));

  return candidates.flatMap((line, index) => {
    const source = tryCreateSource(line, index);
    return source ? [source] : [];
  });
}
