from __future__ import annotations

import re
from dataclasses import dataclass


GOLDEN_RESEARCH_PROMPTS = [
    "AI agents market size, pricing, adoption, and deployment challenges in 2026",
    "Compare LangGraph, CrewAI, and AutoGen for enterprise agent orchestration",
    "India EV charging infrastructure market outlook, policy support, and barriers",
    "Clinical evidence and guideline-backed lifestyle changes after a heart attack",
    "EU AI Act compliance timeline and practical obligations for SaaS companies",
    "Open-source vector databases comparison for RAG applications",
    "NVIDIA data center business risks, competitors, and growth outlook",
    "Global humanoid robotics market, leading companies, and commercialization barriers",
]


@dataclass(frozen=True)
class ReportQualityScore:
    score: int
    section_count: int
    citation_count: int
    distinct_reference_count: int
    numeric_evidence_count: int
    has_table: bool
    has_timeline_or_chronology: bool
    unsupported_reference_count: int


def score_report_quality(report: str, structured_references: list[dict] | None = None) -> ReportQualityScore:
    headings = re.findall(r"^##\s+(.+)$", report, flags=re.MULTILINE)
    body_headings = [
        heading
        for heading in headings
        if heading.lower() not in {"executive summary", "references", "conclusion", "conclusion & future outlook"}
    ]
    citation_ids = set(re.findall(r"\[(E\d+)\]", report))
    reference_urls = set(re.findall(r"https?://[^\s)]+", report))
    structured_urls = {str(reference.get("url", "")) for reference in structured_references or [] if reference.get("url")}
    numeric_evidence = re.findall(r"\b(?:20\d{2}|\d+(?:\.\d+)?%|\$?\d+(?:\.\d+)?\s*(?:billion|million|trillion|k|m|bn)?)\b", report, flags=re.IGNORECASE)
    has_table = bool(re.search(r"^\|.+\|\s*$", report, flags=re.MULTILINE))
    has_timeline = bool(re.search(r"\b(timeline|chronology|forecast|outlook|by 20\d{2}|from 20\d{2})\b", report, flags=re.IGNORECASE))
    unsupported_reference_count = len(reference_urls - structured_urls) if structured_urls else 0

    score = 0
    score += min(20, len(body_headings) * 3)
    score += min(20, len(citation_ids) * 2)
    score += min(20, max(len(reference_urls), len(structured_urls)) * 2)
    score += min(15, len(numeric_evidence))
    score += 10 if has_table else 0
    score += 10 if has_timeline else 0
    score -= min(15, unsupported_reference_count * 3)

    return ReportQualityScore(
        score=max(0, min(100, score)),
        section_count=len(body_headings),
        citation_count=len(citation_ids),
        distinct_reference_count=max(len(reference_urls), len(structured_urls)),
        numeric_evidence_count=len(numeric_evidence),
        has_table=has_table,
        has_timeline_or_chronology=has_timeline,
        unsupported_reference_count=unsupported_reference_count,
    )
