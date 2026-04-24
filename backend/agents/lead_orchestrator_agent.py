"""
Lead Orchestrator — batches research work and routes every model call through ModelRouter.
"""

import operator
import re
from typing import Annotated, TypedDict
from urllib.parse import urlsplit

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from agents.sub_agent import create_sub_agent_graph, normalize_section_tag
from services.config import get_settings
from services.model_router import RequestBudget
from services.runtime import get_model_router, get_settings_instance
from services.source_quality import (
    compute_authority_score,
    fuzzy_token_overlap,
    infer_source_type,
    is_blocked_reference_url,
    is_generic_low_signal_result,
    is_low_value_reference_url,
    is_reference_usable,
    looks_like_homepage,
    query_tokens,
)


class OrchestratorState(TypedDict):
    thread_id: str
    original_query: str
    depth_profile: str
    research_contract: dict
    depth_budget: dict
    research_plan: list[str]
    required_sections: list[str]
    pending_tasks: list[str]
    current_batch: list[str]
    human_feedback: str
    findings: Annotated[list[dict], operator.add]
    evidence_cards: Annotated[list[dict], operator.add]
    sources: Annotated[list[str], operator.add]
    discovered_sources: Annotated[list[str], operator.add]
    coverage_tags: Annotated[list[str], operator.add]
    completed_tasks: Annotated[list[str], operator.add]
    gaps: list[str]
    quality_summary: str
    evaluation_rounds: int
    targeted_gap_rounds: int
    outline_sections: list[str]
    section_packets: list[dict]
    priority_sections: list[str]
    section_drafts: dict[str, str]
    section_verifications: dict[str, dict]
    report_reference_urls: list[str]
    structured_references: list[dict]
    final_report: str


class ResearchContract(TypedDict):
    query_type: str
    must_answer_questions: list[str]
    required_evidence_types: list[str]
    depth_requirements: dict
    report_template: list[str]


class DepthBudget(TypedDict):
    max_worker_tasks: int
    max_gap_rounds: int
    max_verifier_sections: int
    max_repair_passes: int
    max_priority_expansions: int


class SectionPacket(TypedDict):
    section: str
    core_question: str
    thesis: str
    importance_score: int
    readiness_score: int
    supporting_claims: list[str]
    contradictions_or_uncertainties: list[str]
    selected_cards: list[dict]
    best_evidence_cards: list[dict]
    selected_findings: list[dict]
    distinct_source_count: int
    quantitative_fact_count: int
    reference_urls: list[str]
    citation_urls: list[str]
    missing_evidence: list[str]
    required_tables: list[str]
    required_elements: list[str]
    ready_for_draft: bool


class DecomposePlan(BaseModel):
    tasks: list[str] = Field(
        description="Highly specific research tasks that isolated workers can handle independently.",
    )
    required_sections: list[str] = Field(
        description="The sections a strong final report must cover.",
    )


class EvaluateGaps(BaseModel):
    is_complete: bool = Field(description="True when the collected evidence is deep enough to answer the query.")
    gaps: list[str] = Field(description="Specific missing questions that need another round of research.")
    missing_sections: list[str] = Field(description="Important report sections that still need better evidence.")
    quality_summary: str = Field(description="A short explanation of the current evidence quality and what is still weak.")


class OutlinePlan(BaseModel):
    sections: list[str] = Field(description="The final report sections in the order they should appear.")


class SectionVerificationResult(BaseModel):
    passes: bool = Field(description="True when the section is well-supported and needs no repair.")
    issues: list[str] = Field(description="Specific quality, citation, or evidence issues found in the section.")
    repair_instructions: list[str] = Field(description="Concrete repair instructions for the section writer.")
    unsupported_claims: list[str] = Field(description="Claims that are not clearly supported by the evidence brief.")
    missing_elements: list[str] = Field(description="Missing table, timeline, comparison, uncertainty, or citation elements.")


def _dedupe_items(items: list[str], limit: int | None = None) -> list[str]:
    unique_items: list[str] = []
    seen = set()
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique_items.append(cleaned)
        if limit is not None and len(unique_items) >= limit:
            break
    return unique_items


def _dedupe_urls(items: list[str]) -> list[str]:
    return _dedupe_items(items, None)


def _classify_query_type(query: str) -> str:
    lowered = query.lower()
    tokens = _section_tokens(query)
    if tokens & {"medical", "clinical", "patient", "treatment", "therapy", "cardiac", "diagnosis", "symptom", "diet", "health"}:
        return "medical"
    if tokens & {"policy", "regulation", "regulatory", "law", "legal", "compliance", "government", "tax", "subsidy"}:
        return "policy"
    if tokens & {"company", "startup", "vendor", "competitor", "revenue", "valuation", "financial", "earnings"}:
        return "company"
    if " vs " in lowered or " versus " in lowered or tokens & {"compare", "comparison", "alternatives"}:
        return "comparison"
    if tokens & {"architecture", "benchmark", "benchmarks", "api", "model", "models", "framework", "database", "latency", "throughput", "technical"}:
        return "technical"
    if tokens & {"market", "industry", "pricing", "adoption", "growth", "forecast", "share", "landscape", "competition"}:
        return "market"
    return "general"


def _report_template_for_query_type(query_type: str) -> list[str]:
    templates = {
        "market": ["Market size and growth", "Demand and adoption", "Competitive landscape", "Pricing and business models", "Regulation and policy", "Risks and outlook"],
        "technical": ["Current architecture", "Benchmarks and performance", "Tooling and ecosystem", "Implementation tradeoffs", "Limitations and risks", "Future direction"],
        "medical": ["Guideline summary", "Evidence quality", "Patient-specific implications", "Risks and cautions", "Monitoring and follow-up", "Open questions"],
        "policy": ["Current policy position", "Stakeholders and incentives", "Timeline and enforcement", "Economic and social tradeoffs", "Risks and gaps", "Outlook"],
        "company": ["Business overview", "Products and positioning", "Financial and growth signals", "Competitive landscape", "Risks", "Outlook"],
        "comparison": ["Evaluation criteria", "Option-by-option comparison", "Strengths and weaknesses", "Cost and implementation tradeoffs", "Risks", "Recommendation"],
        "general": ["Context", "Key evidence", "Current state", "Tradeoffs", "Risks and uncertainty", "Recommendations"],
    }
    return templates.get(query_type, templates["general"])


def _required_evidence_types_for_query_type(query_type: str) -> list[str]:
    evidence_types = {
        "market": ["recent statistic", "primary/company source", "pricing or adoption data", "reputable industry analysis", "forecast or timeline"],
        "technical": ["official documentation", "benchmark or measurement", "paper/standard/source code", "implementation example", "limitation evidence"],
        "medical": ["clinical guideline", "journal or evidence review", "hospital/authority source", "risk or contraindication evidence", "follow-up guidance"],
        "policy": ["law/regulation/policy document", "government source", "stakeholder evidence", "timeline/enforcement detail", "economic impact evidence"],
        "company": ["company source", "financial filing or announcement", "customer/product evidence", "competitor evidence", "risk evidence"],
        "comparison": ["direct comparison data", "official docs", "benchmark/pricing evidence", "case example", "risk evidence"],
        "general": ["authoritative source", "recent statistic", "counterpoint", "example", "practical implication"],
    }
    return evidence_types.get(query_type, evidence_types["general"])


def _build_must_answer_questions(query: str, query_type: str, required_sections: list[str]) -> list[str]:
    template_questions = {
        "market": [
            "What is the current size, growth rate, and direction of the market?",
            "Which segments, regions, or use cases are driving adoption?",
            "Who are the important competitors or stakeholders, and how do they differ?",
            "What pricing, business model, or investment signals matter most?",
            "What risks, constraints, or uncertainties could change the outlook?",
        ],
        "technical": [
            "What architecture, methods, or standards define the current state?",
            "What benchmarks or implementation evidence separates strong options from weak ones?",
            "What tradeoffs affect performance, cost, reliability, and maintainability?",
            "What limitations, failure modes, or open problems remain?",
            "What practical implementation guidance follows from the evidence?",
        ],
        "medical": [
            "What do reputable guidelines or reviews recommend?",
            "How strong is the evidence behind each recommendation?",
            "What risks, contraindications, or monitoring needs should be surfaced?",
            "Where does guidance differ or remain uncertain?",
            "What practical next steps are supported by the evidence?",
        ],
        "policy": [
            "What is the current legal or policy position?",
            "Who is affected and what incentives or constraints shape behavior?",
            "What timeline, enforcement, or compliance details matter?",
            "What tradeoffs and risks are supported by evidence?",
            "What future changes should readers watch?",
        ],
        "company": [
            "What does the company do and where is it positioned?",
            "What product, customer, or revenue signals matter most?",
            "How does it compare with competitors?",
            "What risks or execution gaps are visible?",
            "What outlook is defensible from the evidence?",
        ],
        "comparison": [
            "What criteria should be used to compare the options?",
            "Where does each option clearly win or lose?",
            "What cost, implementation, or risk tradeoffs matter?",
            "What evidence is strongest or weakest?",
            "What recommendation follows for the likely user context?",
        ],
        "general": [
            "What are the most important facts needed to answer the query?",
            "What evidence supports the main conclusion?",
            "What counterpoints or uncertainty should be acknowledged?",
            "What practical implications follow?",
            "What should be researched further if evidence is thin?",
        ],
    }
    section_questions = [f"What does the evidence show about {section}?" for section in required_sections[:5]]
    return _dedupe_items(template_questions.get(query_type, template_questions["general"]) + section_questions, 10)


def _build_research_contract(query: str, required_sections: list[str]) -> ResearchContract:
    settings = get_settings()
    query_type = _classify_query_type(query)
    requires_numeric = query_type in {"market", "company", "comparison"} or bool(
        _section_tokens(query) & {"pricing", "growth", "forecast", "benchmark", "adoption", "market", "size"}
    )
    freshness_required = query_type in {"market", "policy", "company", "technical"} or "latest" in query.lower()
    return {
        "query_type": query_type,
        "must_answer_questions": _build_must_answer_questions(query, query_type, required_sections),
        "required_evidence_types": _required_evidence_types_for_query_type(query_type),
        "depth_requirements": {
            "min_body_sections": settings.min_body_sections_default,
            "min_distinct_sources": max(settings.min_distinct_sources_for_report, settings.min_body_sections_default + 1),
            "min_evidence_cards": max(
                settings.min_evidence_cards_for_report,
                settings.min_body_sections_default * settings.min_evidence_cards_per_draftable_section,
            ),
            "requires_numeric_evidence": requires_numeric,
            "freshness_required": freshness_required,
            "source_diversity_required": query_type in {"market", "medical", "policy", "technical", "company"},
        },
        "report_template": _report_template_for_query_type(query_type),
    }


def _build_depth_budget(contract: ResearchContract) -> DepthBudget:
    settings = get_settings()
    return {
        "max_worker_tasks": settings.max_initial_tasks,
        "max_gap_rounds": min(settings.max_gap_rounds, 1),
        "max_verifier_sections": settings.max_verifier_sections,
        "max_repair_passes": settings.max_repair_passes,
        "max_priority_expansions": min(settings.priority_section_count, settings.max_priority_expansions),
    }


def _format_research_contract(contract: dict) -> str:
    if not contract:
        return "No research contract was generated."
    questions = "\n".join(f"- {question}" for question in contract.get("must_answer_questions", [])[:10])
    evidence_types = ", ".join(contract.get("required_evidence_types", []))
    requirements = contract.get("depth_requirements", {})
    return (
        f"Query type: {contract.get('query_type', 'general')}\n"
        f"Required evidence types: {evidence_types or 'authoritative evidence'}\n"
        f"Depth requirements: {requirements}\n"
        f"Must-answer questions:\n{questions or '- None'}"
    )


def _worker_task_with_contract(task: str, contract: dict) -> str:
    evidence_types = ", ".join(contract.get("required_evidence_types", [])[:4])
    query_type = contract.get("query_type", "general")
    source_hint = {
        "market": "Include one query targeting official/company sources, pricing, adoption, forecast, PDF, or 2025/2026 data.",
        "technical": "Include one query targeting official docs, benchmark, paper, standard, GitHub, or implementation evidence.",
        "medical": "Include one query targeting guideline, journal, hospital, NIH, PubMed, or systematic review evidence.",
        "policy": "Include one query targeting government, law, regulation, policy document, enforcement, or timeline evidence.",
        "company": "Include one query targeting company announcements, filings, pricing, customer evidence, or competitors.",
        "comparison": "Include one query targeting direct comparison, benchmark, pricing, docs, or implementation tradeoffs.",
        "general": "Include one query targeting authoritative or primary evidence.",
    }.get(query_type, "Include one query targeting authoritative or primary evidence.")
    return (
        f"{task}\n\n"
        "Research contract context:\n"
        f"- Query type: {query_type}\n"
        f"- Required evidence types: {evidence_types or 'authoritative evidence'}\n"
        f"- Search expansion rule: run one broad query, one source-type query, and one gap query only if early results are weak. {source_hint}"
    )


def _is_authoritative_evidence_card(card: dict) -> bool:
    try:
        score = int(card.get("authority_score", 0))
    except (TypeError, ValueError):
        score = 0
    return score >= 8 or card.get("source_type") in {"guideline", "government", "journal", "academic", "hospital", "nonprofit"}


def _card_query_relevance_score(query: str, card: dict) -> int:
    if not query:
        return 0
    query_token_set = _section_tokens(query)
    card_text = " ".join(
        [
            str(card.get("section_tag", "")),
            str(card.get("claim", "")),
            str(card.get("source_title", "")),
            str(card.get("excerpt", ""))[:220],
        ]
    )
    overlap = fuzzy_token_overlap(query_token_set, card_text)
    # Penalize cards with zero query overlap — likely off-topic
    if query_token_set and overlap == 0:
        return -5
    return overlap


def _is_card_relevant_to_query(query: str, card: dict, threshold: float = 0.10) -> bool:
    """Check if an evidence card has minimum topical overlap with the query.
    
    Cards with <10% query token overlap are considered off-topic.
    Uses fuzzy matching (stemming + substring containment) to avoid
    false negatives from natural language inflections.
    """
    query_token_set = _section_tokens(query)
    if not query_token_set:
        return True
    card_text = " ".join([
        str(card.get("claim", "")),
        str(card.get("excerpt", "")),
        str(card.get("source_title", "")),
        str(card.get("section_tag", "")),
    ])
    overlap = fuzzy_token_overlap(query_token_set, card_text)
    return (overlap / len(query_token_set)) >= threshold


def _curate_evidence_cards(evidence_cards: list[dict], scraped_sources: list[str], query: str = "") -> list[dict]:
    from services.source_quality import match_scraped_source

    allowed_sources = set(source for source in scraped_sources if is_reference_usable(source))
    curated: list[dict] = []
    seen = set()
    query_token_set = _section_tokens(query) if query else set()
    for card in evidence_cards:
        source_url = str(card.get("source_url", "")).strip()
        claim = str(card.get("claim", "")).strip()
        if not source_url or not claim:
            continue
        matched, representative = match_scraped_source(source_url, list(allowed_sources))
        verification_status = "verified" if matched else "unverified"
        if matched and representative:
            source_url = representative
        elif source_url not in allowed_sources and not is_reference_usable(source_url):
            # Keep behavior conservative for clearly unusable references.
            continue
        if is_blocked_reference_url(source_url):
            continue
        source_type = card.get("source_type") or infer_source_type(source_url)
        authority_score = int(card.get("authority_score", compute_authority_score(source_url, source_type)))
        if is_low_value_reference_url(source_url) and authority_score < 8:
            continue
        if looks_like_homepage(source_url) and authority_score < 9:
            continue
        source_title = str(card.get("source_title", "")).strip() or source_url
        excerpt = str(card.get("excerpt", "")).strip()[:450]
        if is_generic_low_signal_result(query, source_title, excerpt, source_url):
            continue
        # Query relevance: score + demote rather than drop (avoid false positives)
        overlap = fuzzy_token_overlap(query_token_set, f"{claim} {excerpt} {source_title}") if query_token_set else 0
        query_relevance = (overlap / len(query_token_set)) if query_token_set else 1.0
        cleaned_partial = {
            "claim": claim,
            "excerpt": excerpt,
            "source_title": source_title,
            "section_tag": normalize_section_tag(card.get("section_tag", "")),
            "query_relevance": round(float(query_relevance), 3),
        }
        cleaned = {
            **card,
            "source_url": source_url,
            "claim": claim,
            "excerpt": excerpt,
            "source_title": source_title,
            "source_type": source_type,
            "authority_score": authority_score,
            "section_tag": cleaned_partial["section_tag"],
            "verification_status": str(card.get("verification_status") or verification_status),
            "query_relevance": cleaned_partial["query_relevance"],
        }
        key = (cleaned["section_tag"], source_url, cleaned["claim"].lower())
        if key in seen:
            continue
        seen.add(key)
        curated.append(cleaned)
    curated.sort(
        key=lambda card: (
            _card_query_relevance_score(query, card),
            int(card.get("authority_score", 0)),
            float(card.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    for index, card in enumerate(curated, start=1):
        card["evidence_id"] = str(card.get("evidence_id") or f"E{index}")
    return curated


def _curated_sources(state: OrchestratorState) -> list[str]:
    raw_sources = _dedupe_urls(
        [source for source in state.get("sources", []) if source]
        + [card.get("source_url", "") for card in state.get("evidence_cards", []) if card.get("source_url")]
        + [finding.get("source_url", "") for finding in state.get("findings", []) if finding.get("source_url")]
    )
    return [source for source in raw_sources if is_reference_usable(source) and not is_blocked_reference_url(source)]


def _build_structured_references(
    evidence_cards: list[dict],
    reference_urls: list[str] | None = None,
    limit: int = 18,
) -> list[dict]:
    unique_cards_by_url: dict[str, dict] = {}
    for card in evidence_cards:
        source_url = str(card.get("source_url", "")).strip()
        if source_url and source_url not in unique_cards_by_url:
            unique_cards_by_url[source_url] = card

    if reference_urls:
        unique_cards = [
            unique_cards_by_url[url]
            for url in _dedupe_urls(reference_urls)
            if url in unique_cards_by_url
        ][:limit]
    else:
        # Prefer verified references first; fall back to unverified if needed.
        verified: list[dict] = []
        unverified: list[dict] = []
        for card in unique_cards_by_url.values():
            status = str(card.get("verification_status", "")).lower()
            if status == "verified":
                verified.append(card)
            else:
                unverified.append(card)
        unique_cards = (verified + unverified)[:limit]

    structured: list[dict] = []
    for index, card in enumerate(unique_cards, start=1):
        url = str(card.get("source_url", "")).strip()
        if not url:
            continue
        try:
            hostname = urlsplit(url).netloc.replace("www.", "")
        except Exception:
            hostname = ""
        structured.append(
            {
                "id": f"R{index}",
                "evidence_ids": [str(card.get("evidence_id", ""))] if card.get("evidence_id") else [],
                "title": str(card.get("source_title", "Untitled source")).strip() or "Untitled source",
                "url": url,
                "hostname": hostname,
                "source_type": card.get("source_type", "unknown"),
                "verification_status": str(card.get("verification_status", "unverified")).lower(),
            }
        )
    return structured


def _build_references_section(evidence_cards: list[dict], reference_urls: list[str] | None = None, limit: int = 18) -> str:
    structured_references = _build_structured_references(evidence_cards, reference_urls=reference_urls, limit=limit)

    if not structured_references:
        return "## References\n\n1. No references were available."

    lines = ["## References", ""]
    for index, reference in enumerate(structured_references, start=1):
        title = reference.get("title", "Untitled source")
        url = reference.get("url", "")
        source_type = reference.get("source_type", "unknown")
        status = str(reference.get("verification_status", "")).lower()
        status_note = "" if status == "verified" else "; unverified"
        reference_id = reference.get("id", f"R{index}")
        evidence_ids = ", ".join(reference.get("evidence_ids", []))
        evidence_note = f"; evidence={evidence_ids}" if evidence_ids else ""
        lines.append(f"{index}. [{title}]({url}) ({reference_id}; {source_type}{status_note}{evidence_note})")
    return "\n".join(lines)


def _replace_references_section(report: str, reference_section: str) -> str:
    marker = "\n## References"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip()
    return f"{report.rstrip()}\n\n{reference_section}\n"


def _section_to_source_map(evidence_cards: list[dict]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for card in evidence_cards:
        tag = normalize_section_tag(card.get("section_tag", ""))
        url = card.get("source_url", "").strip()
        if not tag or not url:
            continue
        mapping.setdefault(tag, set()).add(url)
    return mapping


def _find_missing_sections(required_sections: list[str], evidence_cards: list[dict]) -> list[str]:
    settings = get_settings()
    min_sources = max(settings.min_sources_per_section, settings.min_distinct_sources_per_draftable_section)
    source_map = _section_to_source_map(evidence_cards)
    missing = []
    for section in required_sections:
        normalized = normalize_section_tag(section)
        support_count = len(source_map.get(normalized, set()))
        if support_count < min_sources:
            missing.append(section)
    return missing


def _format_findings(findings: list[dict], limit: int = 24) -> str:
    lines: list[str] = []
    for finding in findings[:limit]:
        fact = finding.get("fact", "Unknown fact")
        source = finding.get("source_url", "Unknown source")
        confidence = finding.get("confidence", 0.5)
        lines.append(f"- {fact} (Source: {source}; Confidence: {confidence})")
    return "\n".join(lines) if lines else "No findings were collected."


def _format_evidence_cards(evidence_cards: list[dict], limit: int = 24) -> str:
    lines: list[str] = []
    for card in evidence_cards[:limit]:
        evidence_id = card.get("evidence_id", "?")
        claim = card.get("claim", "Unknown claim")
        excerpt = str(card.get("excerpt", ""))[:220]
        source_title = card.get("source_title", "Unknown source")
        source_url = card.get("source_url", "")
        tag = normalize_section_tag(card.get("section_tag", "general"))
        source_type = card.get("source_type", "unknown")
        authority_score = card.get("authority_score", 0)
        lines.append(
            f"- {evidence_id} [{tag}] {claim} | {source_title} | {source_type} | authority={authority_score} | {source_url} | excerpt={excerpt}"
        )
    return "\n".join(lines) if lines else "No structured evidence cards were collected."


def _section_tokens(value: str) -> set[str]:
    return set(query_tokens(value.replace("_", " ")))


def _query_complexity_score(query: str, required_sections: list[str]) -> int:
    lowered = query.lower()
    broad_markers = sum(
        1
        for marker in (
            "market",
            "industry",
            "global",
            "ecosystem",
            "landscape",
            "pricing",
            "adoption",
            "competition",
            "deployment",
            "policy",
            "regulation",
            "forecast",
            "trend",
            "challenge",
            "risk",
            "opportunity",
            "compare",
            "versus",
            "vs",
        )
        if marker in lowered
    )
    return len(query_tokens(query)) + len(required_sections) * 2 + broad_markers


def _card_has_quantitative_signal(card: dict) -> bool:
    text = f"{card.get('claim', '')} {card.get('excerpt', '')}"
    return bool(re.search(r"\d", text))


def _section_requires_quantitative_support(section: str) -> bool:
    return bool(
        _section_tokens(section)
        & {
            "market",
            "size",
            "pricing",
            "price",
            "prices",
            "cost",
            "costs",
            "growth",
            "adoption",
            "revenue",
            "ranking",
            "rankings",
            "benchmark",
            "benchmarks",
            "timeline",
            "timelines",
            "forecast",
            "forecasts",
            "investment",
            "investments",
            "funding",
            "deployment",
            "deployments",
        }
    )


def _section_required_elements(section: str, selected_cards: list[dict]) -> list[str]:
    elements = [
        "why this section matters",
        "concrete facts and numbers",
        "comparison or tradeoff",
        "uncertainty or contradiction",
        "implication or recommendation",
    ]
    if _section_requires_quantitative_support(section):
        elements.append("quantitative benchmark or timeline")
    elif any(re.search(r"\b20\d{2}\b", f"{card.get('claim', '')} {card.get('excerpt', '')}") for card in selected_cards):
        elements.append("chronology or timeline")
    if _section_tokens(section) & {"risk", "risks", "challenge", "challenges", "barrier", "barriers", "limitation", "limitations"}:
        elements.append("risks and mitigation")
    return elements


def _section_relevance_score(section: str, card: dict) -> int:
    normalized_section = normalize_section_tag(section)
    card_tag = normalize_section_tag(card.get("section_tag", ""))
    score = 0

    if card_tag == normalized_section:
        score += 8
    elif card_tag not in {"", "general"} and (card_tag in normalized_section or normalized_section in card_tag):
        score += 4

    section_tokens = _section_tokens(section) or _section_tokens(normalized_section)
    card_tokens = _section_tokens(
        " ".join(
            [
                card_tag,
                str(card.get("claim", "")),
                str(card.get("source_title", "")),
                str(card.get("excerpt", ""))[:160],
            ]
        )
    )
    score += len(section_tokens & card_tokens) * 2
    return score


def _select_section_evidence(section: str, evidence_cards: list[dict], limit: int = 10) -> list[dict]:
    ranked: list[tuple[int, int, float, dict]] = []
    for card in evidence_cards:
        relevance = _section_relevance_score(section, card)
        if relevance <= 0:
            continue
        ranked.append(
            (
                relevance,
                int(card.get("authority_score", 0)),
                float(card.get("confidence", 0.0)),
                card,
            )
        )

    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [card for _, _, _, card in ranked[:limit]]


def _select_section_findings(section: str, findings: list[dict], evidence_cards: list[dict], limit: int = 8) -> list[dict]:
    section_sources = {str(card.get("source_url", "")).strip() for card in evidence_cards if card.get("source_url")}
    section_tokens = _section_tokens(section)
    ranked: list[tuple[int, float, dict]] = []

    for finding in findings:
        source_url = str(finding.get("source_url", "")).strip()
        fact = str(finding.get("fact", ""))
        score = 0
        if source_url and source_url in section_sources:
            score += 4
        score += len(section_tokens & _section_tokens(fact)) * 2
        if score <= 0:
            continue
        ranked.append((score, float(finding.get("confidence", 0.0)), finding))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [finding for _, _, finding in ranked[:limit]]


def _extract_citation_labels(content: str) -> set[str]:
    labels = set()
    for label in re.findall(r"\[([^\]]+)\]", content):
        cleaned = re.sub(r"\s+", " ", label.strip().lower())
        if cleaned:
            labels.add(cleaned)
    return labels


def _select_reference_urls_for_section(section_content: str, selected_cards: list[dict], fallback_limit: int = 2) -> list[str]:
    citation_labels = _extract_citation_labels(section_content)
    explicit_matches: list[str] = []

    for card in selected_cards:
        source_url = str(card.get("source_url", "")).strip()
        title = re.sub(r"\s+", " ", str(card.get("source_title", "")).strip().lower())
        evidence_id = str(card.get("evidence_id", "")).strip().lower()
        if not source_url or not title:
            continue
        if any(
            (evidence_id and evidence_id == label.lower())
            or title == label
            or title in label
            or label in title
            for label in citation_labels
        ):
            explicit_matches.append(source_url)

    if explicit_matches:
        return _dedupe_urls(explicit_matches)

    return _dedupe_urls(
        [str(card.get("source_url", "")).strip() for card in selected_cards[:fallback_limit] if card.get("source_url")]
    )


def _section_support_snapshot(section: str, evidence_cards: list[dict], limit: int = 8) -> dict:
    selected_cards = _select_section_evidence(section, evidence_cards, limit=limit)
    distinct_sources = {
        str(card.get("source_url", "")).strip()
        for card in selected_cards
        if card.get("source_url")
    }
    quantitative_cards = sum(1 for card in selected_cards if _card_has_quantitative_signal(card))
    top_authority = max((int(card.get("authority_score", 0)) for card in selected_cards), default=0)
    return {
        "selected_cards": selected_cards,
        "card_count": len(selected_cards),
        "distinct_sources": len(distinct_sources),
        "quantitative_cards": quantitative_cards,
        "top_authority": top_authority,
    }


def _section_is_ready_for_draft(section: str, evidence_cards: list[dict]) -> bool:
    settings = get_settings()
    snapshot = _section_support_snapshot(section, evidence_cards)
    min_sources = max(settings.min_sources_per_section, settings.min_distinct_sources_per_draftable_section)

    # Graceful degradation under low evidence density:
    # If the whole job has far fewer cards than the deep-mode target, allow a section
    # to become draftable with 1 strong card / 1 source so we can generate a partial report.
    low_evidence_mode = len(evidence_cards) < max(
        settings.min_evidence_cards_for_report,
        settings.min_body_sections_default * settings.min_evidence_cards_per_draftable_section,
    )
    required_cards = settings.min_evidence_cards_per_draftable_section
    required_sources = min_sources
    if low_evidence_mode:
        required_cards = min(required_cards, 1)
        required_sources = min(required_sources, 1)

    if snapshot["card_count"] < required_cards:
        return False
    if snapshot["distinct_sources"] < required_sources:
        return False
    if _section_requires_quantitative_support(section):
        if low_evidence_mode:
            # With very limited evidence, accept 1 card if it has any number/date signal OR strong authority.
            return snapshot["quantitative_cards"] >= 1 or snapshot["top_authority"] >= 8
        return snapshot["quantitative_cards"] >= settings.min_quant_signals_for_numeric_sections

    if snapshot["quantitative_cards"] >= 1 or snapshot["top_authority"] >= 8:
        return True

    if low_evidence_mode:
        # As a last resort, accept a verified card even if it doesn't contain numbers.
        return any(
            str(card.get("verification_status", "")).lower() == "verified"
            for card in snapshot.get("selected_cards", [])
        )

    return False


def _is_front_or_back_matter_section(section: str) -> bool:
    lowered = " ".join(section.lower().replace("&", " and ").split())
    exact_matches = {
        "overview",
        "executive summary",
        "executive summary / overview",
        "executive summary overview",
        "executive overview",
        "conclusion",
        "recommendations",
        "future outlook",
        "conclusion and recommendations",
        "conclusion and future outlook",
    }
    if lowered in exact_matches:
        return True
    return "executive summary" in lowered or lowered.startswith("conclusion")


def _filter_body_sections(sections: list[str], required_sections: list[str]) -> list[str]:
    body_sections = _dedupe_items([section for section in sections if not _is_front_or_back_matter_section(section)])
    if body_sections:
        return body_sections

    required_body_sections = _dedupe_items(
        [section for section in required_sections if not _is_front_or_back_matter_section(section)]
    )
    if required_body_sections:
        return required_body_sections
    return _dedupe_items(sections)


def _sections_overlap(left: str, right: str) -> bool:
    left_normalized = normalize_section_tag(left)
    right_normalized = normalize_section_tag(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    if left_normalized in right_normalized or right_normalized in left_normalized:
        return True
    left_tokens = _section_tokens(left)
    right_tokens = _section_tokens(right)
    overlap = left_tokens & right_tokens
    smaller = min(len(left_tokens), len(right_tokens))
    return smaller > 0 and len(overlap) >= max(2, smaller - 1)


def _target_body_section_count(query: str, required_sections: list[str], evidence_cards: list[dict]) -> int:
    settings = get_settings()
    distinct_sources = len(
        {
            str(card.get("source_url", "")).strip()
            for card in evidence_cards
            if card.get("source_url")
        }
    )
    covered_tags = len(
        {
            normalize_section_tag(card.get("section_tag", ""))
            for card in evidence_cards
            if card.get("section_tag")
        }
    )
    complexity_score = _query_complexity_score(query, required_sections)
    target_count = settings.min_body_sections_default

    if (
        complexity_score >= 18
        and len(evidence_cards) >= settings.min_body_sections_default * settings.min_evidence_cards_per_draftable_section
        and distinct_sources >= settings.min_body_sections_default + 1
        and covered_tags >= settings.min_body_sections_default
    ):
        target_count += 1
    if (
        complexity_score >= 24
        and len(evidence_cards) >= settings.min_body_sections_default * settings.min_evidence_cards_per_draftable_section + 6
        and distinct_sources >= settings.min_body_sections_default + 3
        and covered_tags >= settings.min_body_sections_default + 1
    ):
        target_count += 1

    return max(settings.min_body_sections_default, min(settings.max_body_sections_deep, target_count))


def _section_support_score(section: str, evidence_cards: list[dict], contract: dict | None = None) -> int:
    snapshot = _section_support_snapshot(section, evidence_cards)
    ready_bonus = 100 if _section_is_ready_for_draft(section, evidence_cards) else 0
    contract_bonus = 0
    if contract:
        section_tokens = _section_tokens(section)
        question_tokens = set()
        for question in contract.get("must_answer_questions", []):
            question_tokens.update(_section_tokens(question))
        contract_bonus = len(section_tokens & question_tokens) * 3
    return (
        ready_bonus
        + snapshot["card_count"] * 8
        + snapshot["distinct_sources"] * 10
        + snapshot["quantitative_cards"] * 4
        + snapshot["top_authority"]
        + contract_bonus
    )


def _prioritize_outline_sections(
    sections: list[str],
    evidence_cards: list[dict],
    target_count: int,
    contract: dict | None = None,
) -> list[str]:
    ranked = [
        (index, section, _section_support_score(section, evidence_cards, contract))
        for index, section in enumerate(sections)
    ]
    ranked.sort(key=lambda item: (item[2], -item[0]), reverse=True)

    chosen: list[str] = []
    for _, section, _ in ranked:
        if any(_sections_overlap(section, existing) for existing in chosen):
            continue
        chosen.append(section)
        if len(chosen) >= target_count:
            break

    return [section for section in sections if section in chosen][:target_count]


def _readiness_score_for_section(section: str, selected_cards: list[dict]) -> int:
    settings = get_settings()
    distinct_sources = len({str(card.get("source_url", "")) for card in selected_cards if card.get("source_url")})
    quantitative_count = sum(1 for card in selected_cards if _card_has_quantitative_signal(card))
    verified_count = sum(1 for card in selected_cards if str(card.get("verification_status", "")).lower() == "verified")
    top_authority = max((int(card.get("authority_score", 0)) for card in selected_cards), default=0)
    score = min(35, len(selected_cards) * 10)
    score += min(25, distinct_sources * 10)
    score += min(15, quantitative_count * 5)
    score += min(15, verified_count * 5)
    score += min(10, top_authority)
    if _section_requires_quantitative_support(section) and quantitative_count < settings.min_quant_signals_for_numeric_sections:
        score -= 15
    return max(0, min(100, score))


def _infer_section_core_question(section: str, contract: dict | None) -> str:
    if contract:
        section_tokens = _section_tokens(section)
        ranked = sorted(
            contract.get("must_answer_questions", []),
            key=lambda question: len(section_tokens & _section_tokens(question)),
            reverse=True,
        )
        if ranked and len(section_tokens & _section_tokens(ranked[0])) > 0:
            return ranked[0]
    return f"What does the evidence show about {section}?"


def _infer_section_thesis(section: str, selected_cards: list[dict]) -> str:
    if not selected_cards:
        return f"The evidence for {section} is not yet strong enough for a confident thesis."
    top_claims = [str(card.get("claim", "")).strip() for card in selected_cards[:2] if card.get("claim")]
    return " ".join(top_claims)[:420] if top_claims else f"The strongest available evidence frames {section} as a material part of the answer."


def _infer_missing_evidence(section: str, selected_cards: list[dict], contract: dict | None) -> list[str]:
    missing: list[str] = []
    distinct_sources = len({card.get("source_url", "") for card in selected_cards if card.get("source_url")})
    if distinct_sources < get_settings().min_distinct_sources_per_draftable_section:
        missing.append("more distinct sources")
    if _section_requires_quantitative_support(section) and not any(_card_has_quantitative_signal(card) for card in selected_cards):
        missing.append("specific numbers, dates, rankings, pricing, adoption, or timeline evidence")
    if contract and contract.get("depth_requirements", {}).get("freshness_required"):
        if not any(re.search(r"\b20(2[4-9]|3\d)\b", f"{card.get('claim', '')} {card.get('excerpt', '')}") for card in selected_cards):
            missing.append("fresh recent evidence")
    if not any(str(card.get("verification_status", "")).lower() == "verified" for card in selected_cards):
        missing.append("verified scraped-source citation")
    return _dedupe_items(missing, 5)


def _required_tables_for_section(section: str, selected_cards: list[dict], contract: dict | None) -> list[str]:
    section_tokens = _section_tokens(section)
    tables: list[str] = []
    if section_tokens & {"pricing", "price", "cost", "market", "size", "growth", "benchmark", "comparison", "competitor", "competition"}:
        tables.append("comparison table")
    if section_tokens & {"timeline", "policy", "regulation", "forecast", "outlook"} or any(
        re.search(r"\b20\d{2}\b", f"{card.get('claim', '')} {card.get('excerpt', '')}") for card in selected_cards
    ):
        tables.append("timeline or chronology table")
    if contract and contract.get("query_type") == "comparison":
        tables.append("option-by-option tradeoff table")
    return _dedupe_items(tables, 3)


def _build_section_packet(
    section: str,
    query: str,
    findings: list[dict],
    evidence_cards: list[dict],
    contract: dict | None = None,
) -> SectionPacket:
    selected_cards = _select_section_evidence(section, evidence_cards, limit=12)
    selected_findings = _select_section_findings(section, findings, selected_cards, limit=10)
    distinct_source_count = len(
        {
            str(card.get("source_url", "")).strip()
            for card in selected_cards
            if card.get("source_url")
        }
    )
    quantitative_fact_count = sum(1 for card in selected_cards if _card_has_quantitative_signal(card))
    readiness_score = _readiness_score_for_section(section, selected_cards)
    importance_score = (
        _section_support_score(section, evidence_cards, contract)
        + len(_section_tokens(section) & _section_tokens(query)) * 6
        + (8 if _section_requires_quantitative_support(section) else 0)
    )
    reference_urls = _dedupe_urls(
        [
            str(card.get("source_url", "")).strip()
            for card in selected_cards
            if card.get("source_url")
        ]
    )
    supporting_claims = _dedupe_items([str(card.get("claim", "")).strip() for card in selected_cards if card.get("claim")], 8)
    contradictions_or_uncertainties = [
        "Evidence remains limited or uneven for this section."
    ] if _infer_missing_evidence(section, selected_cards, contract) else []
    missing_evidence = _infer_missing_evidence(section, selected_cards, contract)
    required_tables = _required_tables_for_section(section, selected_cards, contract)
    return {
        "section": section,
        "core_question": _infer_section_core_question(section, contract),
        "thesis": _infer_section_thesis(section, selected_cards),
        "importance_score": importance_score,
        "readiness_score": readiness_score,
        "supporting_claims": supporting_claims,
        "contradictions_or_uncertainties": contradictions_or_uncertainties,
        "selected_cards": selected_cards,
        "best_evidence_cards": selected_cards[:6],
        "selected_findings": selected_findings,
        "distinct_source_count": distinct_source_count,
        "quantitative_fact_count": quantitative_fact_count,
        "reference_urls": reference_urls,
        "citation_urls": reference_urls,
        "missing_evidence": missing_evidence,
        "required_tables": required_tables,
        "required_elements": _section_required_elements(section, selected_cards),
        "ready_for_draft": _section_is_ready_for_draft(section, evidence_cards),
    }


def _build_section_packets(
    sections: list[str],
    query: str,
    findings: list[dict],
    evidence_cards: list[dict],
    contract: dict | None = None,
) -> list[SectionPacket]:
    return [_build_section_packet(section, query, findings, evidence_cards, contract) for section in sections]


def _supported_section_packets(section_packets: list[SectionPacket]) -> list[SectionPacket]:
    return [packet for packet in section_packets if packet.get("ready_for_draft")]


def _select_priority_sections(section_packets: list[SectionPacket]) -> list[str]:
    settings = get_settings()
    ranked = sorted(
        _supported_section_packets(section_packets),
        key=lambda packet: (
            int(packet.get("importance_score", 0)),
            int(packet.get("readiness_score", 0)),
            int(packet.get("quantitative_fact_count", 0)),
            int(packet.get("distinct_source_count", 0)),
        ),
        reverse=True,
    )
    return [str(packet.get("section", "")) for packet in ranked[: min(settings.priority_section_count, settings.max_priority_expansions)]]


def _build_targeted_gap_tasks(section_packets: list[SectionPacket], target_count: int) -> list[str]:
    settings = get_settings()
    supported_count = len(_supported_section_packets(section_packets))
    needed = max(0, target_count - supported_count)
    if needed == 0:
        return []

    unsupported = [
        packet
        for packet in sorted(section_packets, key=lambda item: int(item.get("importance_score", 0)), reverse=True)
        if not packet.get("ready_for_draft")
    ]
    gap_tasks: list[str] = []
    for packet in unsupported[: min(settings.max_gap_tasks_per_round, needed + 1)]:
        section = str(packet.get("section", "")).strip()
        if not section:
            continue
        requirement = "Add specific numbers, dates, rankings, pricing, adoption, or timeline evidence."
        if not _section_requires_quantitative_support(section):
            requirement = "Add multiple authoritative sources and concrete supporting details."
        gap_tasks.append(f"Collect stronger evidence for section '{section}'. {requirement}")
    return gap_tasks


def _format_section_brief(packet: dict) -> str:
    claims = "\n".join(f"- {claim}" for claim in packet.get("supporting_claims", [])[:8]) or "- No supporting claims."
    uncertainties = "\n".join(f"- {item}" for item in packet.get("contradictions_or_uncertainties", [])[:4]) or "- None identified."
    missing = "\n".join(f"- {item}" for item in packet.get("missing_evidence", [])[:5]) or "- None."
    required_tables = ", ".join(packet.get("required_tables", [])) or "None required"
    citations = ", ".join(
        f"{card.get('evidence_id', '?')}={card.get('source_title', 'Unknown source')}"
        for card in packet.get("best_evidence_cards", [])[:8]
    )
    return (
        f"Section: {packet.get('section', '')}\n"
        f"Core question: {packet.get('core_question', '')}\n"
        f"Working thesis: {packet.get('thesis', '')}\n"
        f"Readiness score: {packet.get('readiness_score', 0)}/100\n"
        f"Required tables: {required_tables}\n"
        f"Citation map: {citations or 'No citations'}\n"
        f"Supporting claims:\n{claims}\n"
        f"Uncertainty / contradictions:\n{uncertainties}\n"
        f"Missing evidence:\n{missing}"
    )


def _format_section_briefs(section_packets: list[dict], limit: int = 8) -> str:
    if not section_packets:
        return "No section evidence briefs were built."
    return "\n\n".join(_format_section_brief(packet) for packet in section_packets[:limit])


def _select_sections_for_verification(
    section_drafts: dict[str, str],
    section_packets: list[dict],
    priority_sections: list[str],
) -> list[str]:
    settings = get_settings()
    packet_by_section = {str(packet.get("section", "")): packet for packet in section_packets}
    candidates: list[tuple[int, str]] = []
    for section, content in section_drafts.items():
        packet = packet_by_section.get(section, {})
        score = 0
        if section in priority_sections:
            score += 100
        readiness = int(packet.get("readiness_score", 0))
        if readiness < 70:
            score += 30
        if len(re.findall(r"\d", content)) >= 8:
            score += 20
        if packet.get("missing_evidence"):
            score += 15
        if score > 0:
            candidates.append((score, section))
    candidates.sort(reverse=True)
    return [section for _, section in candidates[: settings.max_verifier_sections]]


def _verification_passes_locally(content: str, packet: dict) -> bool:
    if not content.strip():
        return False
    labels = _extract_citation_labels(content)
    evidence_ids = {str(card.get("evidence_id", "")).lower() for card in packet.get("best_evidence_cards", [])}
    cited_ids = labels & evidence_ids
    if packet.get("best_evidence_cards") and not cited_ids:
        return False
    filler_phrases = ("significant growth", "various factors", "rapidly evolving", "it is important to note")
    if any(phrase in content.lower() for phrase in filler_phrases):
        return False
    if packet.get("required_tables") and "|" not in content:
        return False
    return True


def _condense_section_for_editor(content: str, limit: int = 900) -> str:
    cleaned = content.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0].rstrip() + "..."


def _report_has_deep_evidence_coverage(
    query: str,
    required_sections: list[str],
    evidence_cards: list[dict],
    distinct_sources: list[str],
    authoritative_source_count: int,
) -> bool:
    settings = get_settings()
    target_cards = max(
        settings.min_evidence_cards_for_report,
        settings.min_body_sections_default * settings.min_evidence_cards_per_draftable_section,
    )
    section_like_support = sum(
        1
        for section in _dedupe_items(required_sections, settings.max_body_sections_deep)
        if _section_is_ready_for_draft(section, evidence_cards)
    )
    depth_floor = min(
        settings.min_body_sections_default,
        max(4, len(_dedupe_items(required_sections, settings.max_body_sections_deep))),
    )
    return (
        len(distinct_sources) >= max(settings.min_distinct_sources_for_report, settings.min_body_sections_default + 1)
        and authoritative_source_count >= settings.min_authoritative_sources_for_report
        and len(evidence_cards) >= target_cards
        and section_like_support >= depth_floor
        and _target_body_section_count(query, required_sections, evidence_cards) >= settings.min_body_sections_default
    )


def _build_insufficient_evidence_report(title: str, quality_summary: str, reference_section: str) -> str:
    lines = [
        f"# {title}",
        "",
        "## Executive Summary",
        "",
        "Validated evidence was not strong enough to generate a reliable final report without introducing unsupported claims.",
        "",
        "## Research Status",
        "",
        "- The system preserved only validated evidence and intentionally skipped synthetic fallback citations.",
        "- Additional targeted research is required before drawing confident conclusions.",
    ]
    if quality_summary.strip():
        lines.extend(["", f"Quality summary: {quality_summary.strip()}"])
    lines.extend(["", reference_section])
    return "\n".join(lines)


def _build_key_findings_table(section_packets: list[dict], section_drafts: dict[str, str]) -> str:
    rows = ["## Key Findings", "", "| Area | Finding | Evidence | Strength |", "|---|---|---|---|"]
    for packet in section_packets:
        section = str(packet.get("section", "")).strip()
        if section not in section_drafts:
            continue
        claim = (packet.get("supporting_claims") or [packet.get("thesis", "")])[0]
        evidence_ids = ", ".join(
            str(card.get("evidence_id", "?"))
            for card in packet.get("best_evidence_cards", [])[:3]
            if card.get("evidence_id")
        )
        strength = "Strong" if int(packet.get("readiness_score", 0)) >= 80 else "Moderate" if int(packet.get("readiness_score", 0)) >= 60 else "Limited"
        rows.append(f"| {section} | {str(claim).replace('|', '/')} | {evidence_ids or 'Evidence limited'} | {strength} |")
    if len(rows) == 4:
        return ""
    return "\n".join(rows)


def _build_evidence_strength_note(section_packets: list[dict], structured_references: list[dict]) -> str:
    if not section_packets:
        return ""
    average_readiness = round(sum(int(packet.get("readiness_score", 0)) for packet in section_packets) / len(section_packets))
    verified_count = sum(1 for reference in structured_references if reference.get("verification_status") == "verified")
    source_count = len(structured_references)
    return (
        "## Evidence Strength\n\n"
        f"The evidence base uses {source_count} curated final reference{'s' if source_count != 1 else ''}, "
        f"including {verified_count} verified scraped-source reference{'s' if verified_count != 1 else ''}. "
        f"Average section readiness is {average_readiness}/100. Claims with thinner support are called out as uncertainty inside the relevant section."
    )


def _build_methodology_note(contract: dict, section_packets: list[dict]) -> str:
    query_type = contract.get("query_type", "general") if contract else "general"
    evidence_types = ", ".join(contract.get("required_evidence_types", [])) if contract else "authoritative sources"
    drafted_sections = ", ".join(str(packet.get("section", "")) for packet in section_packets[:8])
    return (
        "## Methodology / Source Notes\n\n"
        f"The report was assembled as a {query_type} research brief. The system prioritized {evidence_types or 'authoritative sources'}, "
        f"built section-level evidence briefs, expanded the highest-value sections, and verified selected high-risk sections before final assembly. "
        f"Drafted body sections: {drafted_sections or 'none'}."
    )


async def decompose_node(state: OrchestratorState):
    settings = get_settings()
    router = get_model_router()
    query = state["original_query"]
    feedback = state.get("human_feedback", "")

    prompt = (
        "You are planning a deep research investigation. Break the user query into highly specific, "
        "self-contained research tasks that can be run independently by worker agents.\n\n"
        "TASK COUNT RULES — decide dynamically based on query complexity:\n"
        "  • Simple/focused query (single topic, clear answer): 2 tasks\n"
        "  • Moderate query (multiple facets, comparison needed): 3 tasks\n"
        f"  • Complex/multi-domain query (broad scope, many stakeholders): up to {settings.max_initial_tasks} tasks\n\n"
        "EFFICIENCY RULES (critical — we have limited API budget):\n"
        "  • Each task must cover MAXIMUM ground — combine related sub-questions into one task\n"
        "  • NO overlapping tasks — every task must investigate a DISTINCT angle\n"
        "  • Favor breadth over depth — workers will dive deeper on their own\n"
        "  • Order tasks by importance (most critical first)\n\n"
        "Also define the essential report sections needed for a thorough final answer."
    )
    if feedback:
        prompt += f"\nIncorporate this human feedback into the revised plan: {feedback}"

    messages = [
        SystemMessage(content="You are the planning brain for a production research system. You must be efficient with resources while ensuring comprehensive coverage."),
        HumanMessage(content=f"User Query: {query}\n\n{prompt}"),
    ]
    response = await router.generate_structured(
        task_type="planner",
        schema=DecomposePlan,
        messages=messages,
        budget=RequestBudget(max_input_chars=get_settings_instance().planner_input_char_budget, max_output_tokens=500),
        trace_id=state["thread_id"],
    )
    safe_tasks = _dedupe_items(response.tasks, settings.max_initial_tasks)
    required_sections = _dedupe_items(
        response.required_sections,
        max(settings.min_body_sections_default + 1, settings.max_initial_tasks + 2),
    )
    research_contract = _build_research_contract(query, required_sections)
    depth_budget = _build_depth_budget(research_contract)
    safe_tasks = safe_tasks[: depth_budget["max_worker_tasks"]]
    pending_tasks = [_worker_task_with_contract(task, research_contract) for task in safe_tasks]

    return {
        "depth_profile": "deep",
        "research_contract": research_contract,
        "depth_budget": depth_budget,
        "research_plan": safe_tasks,
        "required_sections": required_sections,
        "pending_tasks": pending_tasks,
        "current_batch": [],
        "gaps": [],
        "human_feedback": "",
        "quality_summary": "",
        "targeted_gap_rounds": 0,
        "outline_sections": [],
        "section_packets": [],
        "priority_sections": [],
        "section_drafts": {},
        "section_verifications": {},
        "report_reference_urls": [],
        "structured_references": [],
    }


def plan_review_node(state: OrchestratorState):
    return state


def dispatch_tasks_node(state: OrchestratorState):
    settings = get_settings()
    pending_tasks = list(state.get("pending_tasks", []))
    current_batch = pending_tasks[: settings.max_active_sub_agents_per_job]
    remaining = pending_tasks[settings.max_active_sub_agents_per_job :]
    return {
        "current_batch": current_batch,
        "pending_tasks": remaining,
    }


def post_batch_node(state: OrchestratorState):
    return {"current_batch": []}


async def evaluate_node(state: OrchestratorState):
    settings = get_settings()
    router = get_model_router()
    findings = state.get("findings", [])
    evidence_cards = _curate_evidence_cards(state.get("evidence_cards", []), _curated_sources(state), state["original_query"])
    required_sections = state.get("required_sections", [])
    research_contract = state.get("research_contract", {}) or _build_research_contract(state["original_query"], required_sections)

    distinct_sources = _dedupe_urls(
        [source for source in _curated_sources(state) if source]
        + [card.get("source_url", "") for card in evidence_cards if card.get("source_url")]
        + [finding.get("source_url", "") for finding in findings if finding.get("source_url")]
    )
    authoritative_source_count = len(
        {
            card.get("source_url", "")
            for card in evidence_cards
            if card.get("source_url") and _is_authoritative_evidence_card(card)
        }
    )
    missing_sections = _find_missing_sections(required_sections, evidence_cards)
    code_gate_passed = _report_has_deep_evidence_coverage(
        state["original_query"],
        required_sections,
        evidence_cards,
        distinct_sources,
        authoritative_source_count,
    ) and not missing_sections

    messages = [
        SystemMessage(
            content=(
                "You evaluate whether research results are complete and identify only material gaps. "
                "Be strict. Do not approve shallow evidence or reports that would collapse into only a few thin sections."
            )
        ),
        HumanMessage(
            content=(
                f"User Query: {state['original_query']}\n\n"
                f"Depth Profile: {state.get('depth_profile', 'deep')}\n"
                f"Research Contract:\n{_format_research_contract(research_contract)}\n\n"
                f"Required Report Sections:\n{chr(10).join(f'- {section}' for section in required_sections) or '- None provided'}\n\n"
                f"Distinct Sources Collected: {len(distinct_sources)}\n"
                f"Authoritative Source Count: {authoritative_source_count}\n"
                f"Structured Evidence Cards: {len(evidence_cards)}\n"
                f"Missing Sections by Deterministic Check:\n{chr(10).join(f'- {section}' for section in missing_sections) or '- None'}\n\n"
                f"Collected Findings:\n{_format_findings(findings)}\n\n"
                f"Collected Evidence:\n{_format_evidence_cards(evidence_cards)}\n\n"
                "Decide whether the answer is complete. For broad advice or research requests, do not mark complete if the evidence is shallow, repetitive, based on too few distinct or authoritative sources, or insufficient to support at least six strong body sections. "
                "Also check whether the must-answer questions in the research contract can be answered with the current evidence. "
                "If not complete, provide only the most important missing questions and sections. Reject generic market-analysis filler, vague statements, or unsupported competitor claims."
            )
        ),
    ]
    response = await router.generate_structured(
        task_type="evaluator",
        schema=EvaluateGaps,
        messages=messages,
        budget=RequestBudget(max_input_chars=get_settings_instance().planner_input_char_budget, max_output_tokens=350),
        trace_id=state["thread_id"],
    )

    llm_missing_sections = _dedupe_items(response.missing_sections, settings.max_gap_tasks_per_round)
    combined_missing_sections = _dedupe_items(missing_sections + llm_missing_sections, settings.max_gap_tasks_per_round)

    next_gap_rounds = state.get("evaluation_rounds", 0)
    pending_gap_tasks: list[str] = []
    if not (code_gate_passed and response.is_complete and not combined_missing_sections):
        pending_gap_tasks = _dedupe_items(
            [f"Collect stronger evidence for section: {section}" for section in combined_missing_sections]
            + response.gaps,
            settings.max_gap_tasks_per_round,
        )
        if pending_gap_tasks:
            next_gap_rounds += 1

    quality_summary_parts = [
        response.quality_summary.strip(),
        f"distinct_sources={len(distinct_sources)}",
        f"authoritative_sources={authoritative_source_count}",
        f"evidence_cards={len(evidence_cards)}",
        f"depth_profile={state.get('depth_profile', 'deep')}",
        f"query_type={research_contract.get('query_type', 'general')}",
    ]
    if combined_missing_sections:
        quality_summary_parts.append(f"missing_sections={', '.join(combined_missing_sections)}")

    return {
        "gaps": pending_gap_tasks,
        "pending_tasks": [f"Supplemental Gap Research: {gap}" for gap in pending_gap_tasks],
        "evaluation_rounds": next_gap_rounds,
        "quality_summary": " | ".join(part for part in quality_summary_parts if part),
    }


async def build_outline_node(state: OrchestratorState):
    settings = get_settings()
    router = get_model_router()
    required_sections = state.get("required_sections", [])
    evidence_cards = _curate_evidence_cards(state.get("evidence_cards", []), _curated_sources(state), state["original_query"])
    findings = state.get("findings", [])
    research_contract = state.get("research_contract", {}) or _build_research_contract(state["original_query"], required_sections)
    target_section_count = _target_body_section_count(state["original_query"], required_sections, evidence_cards)
    candidate_count = min(settings.max_body_sections_deep + 2, target_section_count + 2)
    messages = [
        SystemMessage(content="You design comprehensive, publication-grade report outlines. Your outlines produce reports comparable to professional research firms and Gemini Deep Research."),
        HumanMessage(
            content=(
                f"User Query: {state['original_query']}\n\n"
                f"Research Contract:\n{_format_research_contract(research_contract)}\n\n"
                f"Report Template Hints:\n{chr(10).join(f'- {section}' for section in research_contract.get('report_template', []))}\n\n"
                f"Required Sections:\n{chr(10).join(f'- {section}' for section in required_sections)}\n\n"
                f"Evidence Snapshot:\n{_format_evidence_cards(evidence_cards, limit=20)}\n\n"
                f"Create {candidate_count} candidate body sections for the report. The report must feel like a professional research document.\n\n"
                "REQUIRED STRUCTURE:\n"
                "- Return BODY sections only. Do not include Executive Summary, Overview, Conclusion, or References.\n"
                "- Prioritize the most decision-useful sections and merge overlapping angles aggressively.\n"
                f"- The final report will keep the best-supported {target_section_count} sections, so propose sections with enough substance to survive that cut.\n"
                "- Prefer deep coverage over broad shallow coverage.\n"
                "- Every section must earn its place by being important and evidence-rich.\n\n"
                "For market/industry topics: include market size, competitive landscape, pricing/business models, \n"
                "infrastructure, policy tailwinds, investment trends, and regional analysis.\n"
                "For technical/scientific topics: include methodology, current state, key findings, applications, limitations.\n"
                "Order sections logically. Each section title should be specific and descriptive, not generic."
            )
        ),
    ]
    response = await router.generate_structured(
        task_type="planner",
        schema=OutlinePlan,
        messages=messages,
        budget=RequestBudget(max_input_chars=get_settings_instance().planner_input_char_budget, max_output_tokens=500),
        trace_id=state["thread_id"],
    )
    body_section_candidates = _filter_body_sections(
        _dedupe_items(response.sections + required_sections, max(candidate_count + 2, settings.min_body_sections_default + 2)),
        required_sections,
    )
    outline_sections = _prioritize_outline_sections(
        body_section_candidates,
        evidence_cards,
        target_section_count,
        research_contract,
    )
    section_packets = _build_section_packets(outline_sections, state["original_query"], findings, evidence_cards, research_contract)
    supported_packets = _supported_section_packets(section_packets)
    priority_sections = _select_priority_sections(section_packets)
    targeted_gap_rounds = state.get("targeted_gap_rounds", 0)
    targeted_gap_tasks: list[str] = []
    quality_summary = state.get("quality_summary", "")

    if len(supported_packets) < settings.min_body_sections_default:
        targeted_gap_tasks = _build_targeted_gap_tasks(section_packets, settings.min_body_sections_default)
        if targeted_gap_tasks and targeted_gap_rounds < settings.max_targeted_gap_rounds_deep:
            targeted_gap_rounds += 1
            quality_summary = " | ".join(
                part
                for part in (
                    quality_summary.strip(),
                    f"supported_sections={len(supported_packets)}/{settings.min_body_sections_default}",
                    f"targeted_gap_round={targeted_gap_rounds}",
                    "status=collecting_section_specific_depth",
                )
                if part
            )
            return {
                "gaps": targeted_gap_tasks,
                "pending_tasks": [f"Supplemental Gap Research: {gap}" for gap in targeted_gap_tasks],
                "targeted_gap_rounds": targeted_gap_rounds,
                "quality_summary": quality_summary,
                "outline_sections": [],
                "section_packets": section_packets,
                "priority_sections": priority_sections,
            }

        # --- Graceful degradation: use whatever supported sections we have ---
        # Instead of giving up entirely, draft the sections we can support.
        # Only truly fail when we have 0-1 draftable sections.
        _MIN_SECTIONS_FOR_PARTIAL_REPORT = 2

        if len(supported_packets) >= _MIN_SECTIONS_FOR_PARTIAL_REPORT:
            # We have enough for a partial but useful report
            supported_sections = [packet["section"] for packet in supported_packets][:target_section_count]
            supported_packets_filtered = [packet for packet in section_packets if packet["section"] in supported_sections]
            priority_sections_filtered = [section for section in priority_sections if section in supported_sections]
            quality_summary = " | ".join(
                part
                for part in (
                    quality_summary.strip(),
                    f"supported_sections={len(supported_packets)}/{settings.min_body_sections_default}",
                    "status=partial_report_with_available_evidence",
                )
                if part
            )
            return {
                "gaps": [],
                "pending_tasks": [],
                "quality_summary": quality_summary,
                "outline_sections": supported_sections,
                "section_packets": supported_packets_filtered,
                "priority_sections": priority_sections_filtered,
            }

        # Truly insufficient — fewer than 2 draftable sections
        quality_summary = " | ".join(
            part
            for part in (
                quality_summary.strip(),
                f"supported_sections={len(supported_packets)}/{settings.min_body_sections_default}",
                "status=insufficient_supported_sections_for_deep_report",
            )
            if part
        )
        return {
            "gaps": [],
            "pending_tasks": [],
            "quality_summary": quality_summary,
            "outline_sections": [],
            "section_packets": section_packets,
            "priority_sections": priority_sections,
        }

    supported_sections = [packet["section"] for packet in supported_packets][:target_section_count]
    supported_packets = [packet for packet in section_packets if packet["section"] in supported_sections]
    priority_sections = [section for section in priority_sections if section in supported_sections]
    return {
        "gaps": [],
        "pending_tasks": [],
        "outline_sections": supported_sections,
        "section_packets": supported_packets,
        "priority_sections": priority_sections,
    }


async def build_evidence_briefs_node(state: OrchestratorState):
    evidence_cards = _curate_evidence_cards(state.get("evidence_cards", []), _curated_sources(state), state["original_query"])
    research_contract = state.get("research_contract", {}) or _build_research_contract(state["original_query"], state.get("required_sections", []))
    refreshed_packets = _build_section_packets(
        state.get("outline_sections", []),
        state["original_query"],
        state.get("findings", []),
        evidence_cards,
        research_contract,
    )
    supported_packets = _supported_section_packets(refreshed_packets)
    priority_sections = _select_priority_sections(supported_packets)
    return {
        "section_packets": supported_packets,
        "priority_sections": priority_sections,
    }


async def draft_sections_node(state: OrchestratorState):
    settings = get_settings()
    router = get_model_router()
    evidence_cards = _curate_evidence_cards(state.get("evidence_cards", []), _curated_sources(state), state["original_query"])
    findings = state.get("findings", [])
    research_contract = state.get("research_contract", {}) or _build_research_contract(state["original_query"], state.get("required_sections", []))
    section_packets = state.get("section_packets", [])
    packet_by_section = {
        str(packet.get("section", "")): packet
        for packet in section_packets
        if str(packet.get("section", "")).strip()
    }
    section_drafts: dict[str, str] = {}

    for section in state.get("outline_sections", []):
        packet = packet_by_section.get(section)
        if packet is None:
            packet = _build_section_packet(section, state["original_query"], findings, evidence_cards, research_contract)
        if not packet.get("ready_for_draft"):
            continue
        selected_cards = list(packet.get("selected_cards", []))
        supporting_findings = list(packet.get("selected_findings", []))

        evidence_text = _format_evidence_cards(selected_cards, limit=12)
        section_brief_text = _format_section_brief(packet)
        required_elements_text = "\n".join(f"- {item}" for item in packet.get("required_elements", []))
        base_target = settings.base_section_word_target
        messages = [
            SystemMessage(
                content=(
                    "You are an expert research writer producing publication-grade report sections. "
                    "Your writing should be comparable to professional consulting reports (McKinsey, BCG) "
                    "and Gemini Deep Research output — detailed, data-rich, and actionable."
                )
            ),
            HumanMessage(
                content=(
                    f"User Query: {state['original_query']}\n\n"
                    f"Research Contract:\n{_format_research_contract(research_contract)}\n\n"
                    f"Section Title: {section}\n\n"
                    f"Section Evidence Brief:\n{section_brief_text}\n\n"
                    f"Relevant Evidence:\n{evidence_text}\n\n"
                    f"Supporting Findings:\n{_format_findings(supporting_findings, limit=10)}\n\n"
                    f"Required Analytical Elements:\n{required_elements_text}\n\n"
                    f"Write a DETAILED Markdown section ({max(700, base_target - 150)}-{base_target + 150} words). Requirements:\n\n"
                    "FORMAT & STRUCTURE:\n"
                    "- Start with 2-3 sentences that directly explain why this section matters.\n"
                    "- Use 3-5 ### subheadings to organize the analysis.\n"
                    "- Include Markdown TABLES where comparing data (players, pricing, metrics, timelines).\n"
                    "- Use bullet lists only when they improve clarity.\n"
                    "- End with a brief implication or decision-oriented takeaway.\n\n"
                    "CONTENT QUALITY:\n"
                    "- Focus on the most important and best-supported insights, not background filler.\n"
                    "- Include ALL specific numbers, dates, company names, and statistics from the evidence.\n"
                    "- Cite evidence inline using evidence IDs in brackets, like [E1].\n"
                    "- Every major claim, number, comparison, and risk statement must cite at least one evidence ID.\n"
                    "- Analyze trends, tradeoffs, and relationships; do not just list facts.\n"
                    "- Compare and contrast when multiple data points exist.\n"
                    "- Call out uncertainty, contradictions, or missing evidence explicitly.\n"
                    "- Include at least one concrete comparison, ranking, chronology, or stakeholder contrast when the evidence allows it.\n"
                    "- NO generic filler phrases like 'significant growth' or 'various factors' — be specific."
                )
            ),
        ]
        response = await router.generate_text(
            task_type="synthesis",
            messages=messages,
            budget=RequestBudget(
                max_input_chars=min(get_settings_instance().synthesis_input_char_budget, 24000),
                max_output_tokens=settings.section_draft_output_tokens,
            ),
            trace_id=state["thread_id"],
        )
        section_drafts[section] = response.content

    return {"section_drafts": section_drafts}


async def expand_priority_sections_node(state: OrchestratorState):
    settings = get_settings()
    router = get_model_router()
    evidence_cards = _curate_evidence_cards(state.get("evidence_cards", []), _curated_sources(state), state["original_query"])
    findings = state.get("findings", [])
    research_contract = state.get("research_contract", {}) or _build_research_contract(state["original_query"], state.get("required_sections", []))
    section_drafts = dict(state.get("section_drafts", {}))
    packet_by_section = {
        str(packet.get("section", "")): packet
        for packet in state.get("section_packets", [])
        if str(packet.get("section", "")).strip()
    }

    for section in state.get("priority_sections", []):
        current_draft = section_drafts.get(section, "").strip()
        if not current_draft:
            continue

        packet = packet_by_section.get(section)
        if packet is None:
            packet = _build_section_packet(section, state["original_query"], findings, evidence_cards, research_contract)
        if not packet.get("ready_for_draft"):
            continue

        selected_cards = list(packet.get("selected_cards", []))
        supporting_findings = list(packet.get("selected_findings", []))
        evidence_text = _format_evidence_cards(selected_cards, limit=12)
        section_brief_text = _format_section_brief(packet)
        required_elements_text = "\n".join(f"- {item}" for item in packet.get("required_elements", []))
        priority_target = settings.priority_section_word_target
        messages = [
            SystemMessage(
                content=(
                    "You deepen an already strong research section without adding unsupported claims. "
                    "Your job is to make the section feel like a true deep-research deliverable: denser, more comparative, "
                    "more explicit about uncertainty, and more useful for decision-making."
                )
            ),
            HumanMessage(
                content=(
                    f"User Query: {state['original_query']}\n\n"
                    f"Research Contract:\n{_format_research_contract(research_contract)}\n\n"
                    f"Section Title: {section}\n\n"
                    f"Section Evidence Brief:\n{section_brief_text}\n\n"
                    f"Current Draft:\n{current_draft}\n\n"
                    f"Relevant Evidence:\n{evidence_text}\n\n"
                    f"Supporting Findings:\n{_format_findings(supporting_findings, limit=10)}\n\n"
                    f"Required Analytical Elements:\n{required_elements_text}\n\n"
                    f"Expand this section to approximately {max(1100, priority_target - 150)}-{priority_target + 150} words.\n\n"
                    "EXPANSION RULES:\n"
                    "- Preserve all validated facts and existing inline citations.\n"
                    "- Use evidence IDs like [E1], [E2], not source titles, for any new citations.\n"
                    "- Add depth, not fluff.\n"
                    "- Strengthen comparisons, chronologies, stakeholder contrasts, and counterpoints where supported.\n"
                    "- Add or improve a Markdown table if it helps compare metrics, vendors, timelines, or tradeoffs.\n"
                    "- Surface uncertainty, contradictory signals, and practical implications explicitly.\n"
                    "- Do not introduce claims that are not grounded in the evidence provided."
                )
            ),
        ]
        response = await router.generate_text(
            task_type="synthesis",
            messages=messages,
            budget=RequestBudget(
                max_input_chars=min(get_settings_instance().synthesis_input_char_budget, 28000),
                max_output_tokens=settings.section_draft_output_tokens,
            ),
            trace_id=state["thread_id"],
        )
        section_drafts[section] = response.content

    return {"section_drafts": section_drafts}


async def verify_sections_node(state: OrchestratorState):
    settings = get_settings()
    router = get_model_router()
    section_drafts = dict(state.get("section_drafts", {}))
    section_packets = state.get("section_packets", [])
    packet_by_section = {
        str(packet.get("section", "")): packet
        for packet in section_packets
        if str(packet.get("section", "")).strip()
    }
    sections_to_verify = _select_sections_for_verification(
        section_drafts,
        section_packets,
        state.get("priority_sections", []),
    )
    section_verifications: dict[str, dict] = {}

    for section in sections_to_verify:
        packet = packet_by_section.get(section, {})
        current_draft = section_drafts.get(section, "")
        if _verification_passes_locally(current_draft, packet):
            section_verifications[section] = {
                "passes": True,
                "issues": [],
                "repair_instructions": [],
                "unsupported_claims": [],
                "missing_elements": [],
            }
            continue

        messages = [
            SystemMessage(
                content=(
                    "You are a strict research-section verifier. Check support, citations, depth, and clarity. "
                    "Be concise and practical; only fail sections for material problems."
                )
            ),
            HumanMessage(
                content=(
                    f"User Query: {state['original_query']}\n\n"
                    f"Section Title: {section}\n\n"
                    f"Section Evidence Brief:\n{_format_section_brief(packet)}\n\n"
                    f"Draft Section:\n{current_draft}\n\n"
                    "Verify these requirements:\n"
                    "- major claims and numbers cite evidence IDs like [E1]\n"
                    "- citations match evidence IDs in the brief\n"
                    "- no generic unsupported filler\n"
                    "- required comparisons, tables, timelines, or uncertainty notes are present when the brief asks for them\n"
                    "- contradictions or missing evidence are acknowledged when relevant"
                )
            ),
        ]
        verification = await router.generate_structured(
            task_type="section_verifier",
            schema=SectionVerificationResult,
            messages=messages,
            budget=RequestBudget(max_input_chars=min(get_settings_instance().synthesis_input_char_budget, 18000), max_output_tokens=500),
            trace_id=state["thread_id"],
        )
        section_verifications[section] = verification.model_dump()

        if verification.passes or settings.max_repair_passes <= 0:
            continue

        repair_messages = [
            SystemMessage(
                content=(
                    "You repair a research report section using only the provided evidence brief and verifier feedback. "
                    "Preserve supported content, remove or qualify unsupported claims, and add missing analytical structure."
                )
            ),
            HumanMessage(
                content=(
                    f"User Query: {state['original_query']}\n\n"
                    f"Section Title: {section}\n\n"
                    f"Section Evidence Brief:\n{_format_section_brief(packet)}\n\n"
                    f"Current Draft:\n{current_draft}\n\n"
                    f"Verifier Issues:\n{chr(10).join(f'- {issue}' for issue in verification.issues) or '- None'}\n\n"
                    f"Repair Instructions:\n{chr(10).join(f'- {item}' for item in verification.repair_instructions) or '- Repair citation/depth issues.'}\n\n"
                    "Return the fully repaired Markdown section only. Use evidence IDs like [E1]."
                )
            ),
        ]
        repaired = await router.generate_text(
            task_type="section_repair",
            messages=repair_messages,
            budget=RequestBudget(max_input_chars=min(get_settings_instance().synthesis_input_char_budget, 22000), max_output_tokens=settings.section_draft_output_tokens),
            trace_id=state["thread_id"],
        )
        section_drafts[section] = repaired.content

    return {
        "section_drafts": section_drafts,
        "section_verifications": section_verifications,
    }


async def final_edit_node(state: OrchestratorState):
    settings = get_settings()
    router = get_model_router()
    sources = _curated_sources(state)
    evidence_cards = _curate_evidence_cards(state.get("evidence_cards", []), sources, state["original_query"])
    section_drafts = state.get("section_drafts", {})
    research_contract = state.get("research_contract", {}) or _build_research_contract(state["original_query"], state.get("required_sections", []))
    section_packets = state.get("section_packets", [])
    packet_by_section = {
        str(packet.get("section", "")): packet
        for packet in section_packets
        if str(packet.get("section", "")).strip()
    }

    report_reference_urls: list[str] = []
    for section, content in section_drafts.items():
        selected_cards = list(packet_by_section.get(section, {}).get("selected_cards", [])) or _select_section_evidence(
            section,
            evidence_cards,
            limit=12,
        )
        report_reference_urls.extend(_select_reference_urls_for_section(content, selected_cards))
    report_reference_urls = _dedupe_urls(report_reference_urls)

    # Programmatic Assembly: We don't want the LLM to rewrite these sections as it will
    # inevitably compress and summarize them. We want the full depth.
    draft_text_formatted = "\n\n".join(f"## {title}\n{content}" for title, content in section_drafts.items())
    structured_references = _build_structured_references(evidence_cards, reference_urls=report_reference_urls)
    reference_section = _build_references_section(evidence_cards, reference_urls=report_reference_urls)

    report_title = state["original_query"].title()
    if len(report_title) > 60:
        report_title = "Deep Research Report"

    if not evidence_cards or not section_drafts:
        return {
            "report_reference_urls": report_reference_urls,
            "structured_references": structured_references,
            "final_report": _build_insufficient_evidence_report(
                report_title,
                state.get("quality_summary", ""),
                reference_section,
            )
        }

    section_digest = "\n\n".join(
        f"## {title}\n{_condense_section_for_editor(content)}"
        for title, content in section_drafts.items()
        if content.strip()
    )
    section_briefs_digest = _format_section_briefs(
        [packet for packet in section_packets if packet.get("section") in section_drafts],
        limit=8,
    )

    # We only ask the LLM to generate the front-matter (Executive Summary)
    # and back-matter (Conclusion) to sandwich the generated sections seamlessly.
    messages = [
        SystemMessage(
            content=(
                "You are an executive research editor. "
                "Your task is to write ONLY the Executive Summary and the Final Conclusion for a large research report. "
                "The core detailed sections have already been written by specialist agents."
            )
        ),
        HumanMessage(
            content=(
                f"User Query: {state['original_query']}\n\n"
                f"Research Contract:\n{_format_research_contract(research_contract)}\n\n"
                f"Section Evidence Briefs:\n{section_briefs_digest}\n\n"
                f"Drafted Body Sections:\n{section_digest}\n\n"
                "Write ONLY two things:\n"
                "1. An 'Executive Summary' (3-4 bullet points and a short summary paragraph highlighting the most critical insights from the evidence).\n"
                "2. A 'Conclusion & Future Outlook' (A strong concluding section with actionable recommendations or future projections based on the evidence).\n\n"
                "Do NOT write the middle sections. Do NOT write an introduction. Focus on high-impact insights.\n"
                "Use EXACTLY these markdown headers so I can parse your output:\n\n"
                "## Executive Summary\n"
                "[Your text here]\n\n"
                "## Conclusion\n"
                "[Your text here]"
            )
        ),
    ]
    response = await router.generate_text(
        task_type="synthesis",
        messages=messages,
        budget=RequestBudget(
            max_input_chars=min(get_settings_instance().synthesis_input_char_budget, 24000),
            max_output_tokens=min(settings.final_report_output_tokens, 2000),
        ),
        trace_id=state["thread_id"],
    )

    # Extract the generated front/back matter
    llm_output = response.content
    exec_summary = ""
    conclusion = ""

    match = re.search(
        r"##\s*Executive Summary\s*(.*?)\s*##\s*Conclusion\b\s*(.*)",
        llm_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        exec_summary = match.group(1).strip()
        conclusion = match.group(2).strip()
    else:
        # Fallback if the LLM didn't format perfectly
        exec_summary = llm_output
        conclusion = "The body sections below contain the validated findings. Any remaining uncertainty should be resolved with targeted follow-up research rather than assumption."

    # Programmatic Report Generation
    key_findings_table = _build_key_findings_table(section_packets, section_drafts)
    evidence_strength_note = _build_evidence_strength_note(
        [packet for packet in section_packets if packet.get("section") in section_drafts],
        structured_references,
    )
    methodology_note = _build_methodology_note(
        research_contract,
        [packet for packet in section_packets if packet.get("section") in section_drafts],
    )
    final_report = f"# {report_title}\n\n"
    final_report += f"## Executive Summary\n\n{exec_summary}\n\n"
    if key_findings_table:
        final_report += f"{key_findings_table}\n\n"
    if evidence_strength_note:
        final_report += f"{evidence_strength_note}\n\n"
    final_report += f"{methodology_note}\n\n"
    final_report += f"{draft_text_formatted}\n\n"

    # Add Research Limitations note for partial reports
    quality_summary = state.get("quality_summary", "")
    if "partial_report" in quality_summary:
        section_count = len(section_drafts)
        final_report += (
            "## Research Limitations\n\n"
            f"This report covers {section_count} validated section{'s' if section_count != 1 else ''} "
            "based on the evidence that met quality and relevance thresholds. "
            "Some planned sections could not be drafted due to insufficient authoritative evidence. "
            "Additional targeted research may strengthen the analysis in under-covered areas.\n\n"
        )

    final_report += f"## Conclusion & Future Outlook\n\n{conclusion}\n\n"
    final_report += f"{reference_section}"

    return {
        "report_reference_urls": report_reference_urls,
        "structured_references": structured_references,
        "final_report": final_report,
    }


def route_human_approval(state: OrchestratorState):
    if state.get("human_feedback"):
        return "decompose_node"
    return "dispatch_tasks_node"


def route_batch_dispatch(state: OrchestratorState):
    sends = []
    for task in state.get("current_batch", []):
        sends.append(
            Send(
                "sub_agent",
                {
                    "trace_id": state["thread_id"],
                    "task": task,
                    "messages": [HumanMessage(content=f"Research the following topic strictly and thoroughly:\n\n{task}")],
                    "working_summary": "",
                    "findings": [],
                    "evidence_cards": [],
                    "sources": [],
                    "discovered_sources": [],
                    "seen_source_urls": [],
                    "coverage_tags": [],
                    "completed_tasks": [],
                    "iterations": 0,
                    "status": "running",
                },
            )
        )
    if sends:
        return sends
    return "evaluate_node"


def route_after_batch(state: OrchestratorState):
    if state.get("pending_tasks"):
        return "dispatch_tasks_node"
    return "evaluate_node"


def route_evaluation(state: OrchestratorState):
    settings = get_settings()
    depth_budget = state.get("depth_budget", {}) or {}
    max_gap_rounds = int(depth_budget.get("max_gap_rounds", settings.max_gap_rounds))
    if state.get("pending_tasks") and state.get("evaluation_rounds", 0) <= max_gap_rounds:
        return "dispatch_tasks_node"
    return "build_outline_node"


def route_after_outline(state: OrchestratorState):
    if state.get("pending_tasks"):
        return "dispatch_tasks_node"
    if state.get("outline_sections"):
        return "build_evidence_briefs_node"
    return "final_edit_node"


def create_lead_orchestrator(checkpointer=None):
    builder = StateGraph(OrchestratorState)

    builder.add_node("decompose_node", decompose_node)
    builder.add_node("plan_review_node", plan_review_node)
    builder.add_node("dispatch_tasks_node", dispatch_tasks_node)
    builder.add_node("sub_agent", create_sub_agent_graph())
    builder.add_node("post_batch_node", post_batch_node)
    builder.add_node("evaluate_node", evaluate_node)
    builder.add_node("build_outline_node", build_outline_node)
    builder.add_node("build_evidence_briefs_node", build_evidence_briefs_node)
    builder.add_node("draft_sections_node", draft_sections_node)
    builder.add_node("expand_priority_sections_node", expand_priority_sections_node)
    builder.add_node("verify_sections_node", verify_sections_node)
    builder.add_node("final_edit_node", final_edit_node)

    builder.add_edge(START, "decompose_node")
    builder.add_edge("decompose_node", "plan_review_node")
    builder.add_conditional_edges("plan_review_node", route_human_approval)
    builder.add_conditional_edges("dispatch_tasks_node", route_batch_dispatch)
    builder.add_edge("sub_agent", "post_batch_node")
    builder.add_conditional_edges("post_batch_node", route_after_batch)
    builder.add_conditional_edges("evaluate_node", route_evaluation)
    builder.add_conditional_edges("build_outline_node", route_after_outline)
    builder.add_edge("build_evidence_briefs_node", "draft_sections_node")
    builder.add_edge("draft_sections_node", "expand_priority_sections_node")
    builder.add_edge("expand_priority_sections_node", "verify_sections_node")
    builder.add_edge("verify_sections_node", "final_edit_node")
    builder.add_edge("final_edit_node", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["plan_review_node"],
    )
