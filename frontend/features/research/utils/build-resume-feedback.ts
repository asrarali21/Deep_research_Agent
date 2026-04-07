export function buildResumeFeedback(original: string[], edited: string[], note: string) {
  const removals = original.filter((item) => !edited.includes(item)).map((item) => `- Remove: ${item}`);
  const additions = edited.filter((item) => !original.includes(item)).map((item) => `- Add: ${item}`);
  const trimmedNote = note.trim();

  if (!removals.length && !additions.length && !trimmedNote) {
    return "";
  }

  return ["Revise the research plan using these edits:", ...removals, ...additions, trimmedNote ? `User note: ${trimmedNote}` : ""]
    .filter(Boolean)
    .join("\n");
}
