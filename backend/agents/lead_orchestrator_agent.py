"""
Lead Orchestrator — batches research work and routes every model call through ModelRouter.
"""

import operator
from typing import Annotated, TypedDict

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
    infer_source_type,
    is_blocked_reference_url,
    is_low_value_reference_url,
    is_reference_usable,
    looks_like_homepage,
)


class OrchestratorState(TypedDict):
    thread_id: str
    original_query: str
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
    outline_sections: list[str]
    section_drafts: dict[str, str]
    final_report: str


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


def _is_authoritative_evidence_card(card: dict) -> bool:
    try:
        score = int(card.get("authority_score", 0))
    except (TypeError, ValueError):
        score = 0
    return score >= 8 or card.get("source_type") in {"guideline", "government", "journal", "academic", "hospital", "nonprofit"}


def _curate_evidence_cards(evidence_cards: list[dict], scraped_sources: list[str]) -> list[dict]:
    allowed_sources = set(source for source in scraped_sources if is_reference_usable(source))
    curated: list[dict] = []
    seen = set()
    for card in evidence_cards:
        source_url = str(card.get("source_url", "")).strip()
        claim = str(card.get("claim", "")).strip()
        if not source_url or not claim:
            continue
        if source_url not in allowed_sources:
            continue
        if is_blocked_reference_url(source_url):
            continue
        source_type = card.get("source_type") or infer_source_type(source_url)
        authority_score = int(card.get("authority_score", compute_authority_score(source_url, source_type)))
        if is_low_value_reference_url(source_url) and authority_score < 8:
            continue
        if looks_like_homepage(source_url) and authority_score < 9:
            continue
        cleaned = {
            **card,
            "source_url": source_url,
            "claim": claim,
            "excerpt": str(card.get("excerpt", "")).strip()[:450],
            "source_title": str(card.get("source_title", "")).strip() or source_url,
            "source_type": source_type,
            "authority_score": authority_score,
            "section_tag": normalize_section_tag(card.get("section_tag", "")),
        }
        key = (cleaned["section_tag"], source_url, cleaned["claim"].lower())
        if key in seen:
            continue
        seen.add(key)
        curated.append(cleaned)
    curated.sort(
        key=lambda card: (
            int(card.get("authority_score", 0)),
            float(card.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    return curated


def _curated_sources(state: OrchestratorState) -> list[str]:
    raw_sources = _dedupe_urls(
        [source for source in state.get("sources", []) if source]
        + [card.get("source_url", "") for card in state.get("evidence_cards", []) if card.get("source_url")]
        + [finding.get("source_url", "") for finding in state.get("findings", []) if finding.get("source_url")]
    )
    return [source for source in raw_sources if is_reference_usable(source) and not is_blocked_reference_url(source)]


def _build_references_section(evidence_cards: list[dict], limit: int = 18) -> str:
    unique_cards: list[dict] = []
    seen_urls = set()
    for card in evidence_cards:
        source_url = card.get("source_url", "")
        if not source_url or source_url in seen_urls:
            continue
        seen_urls.add(source_url)
        unique_cards.append(card)
        if len(unique_cards) >= limit:
            break

    if not unique_cards:
        return "## References\n\n1. No validated references were available."

    lines = ["## References", ""]
    for index, card in enumerate(unique_cards, start=1):
        title = card.get("source_title", "Untitled source").strip() or "Untitled source"
        url = card.get("source_url", "")
        source_type = card.get("source_type", "unknown")
        lines.append(f"{index}. [{title}]({url}) ({source_type})")
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
    source_map = _section_to_source_map(evidence_cards)
    missing = []
    for section in required_sections:
        normalized = normalize_section_tag(section)
        support_count = len(source_map.get(normalized, set()))
        if support_count < settings.min_sources_per_section:
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
        claim = card.get("claim", "Unknown claim")
        excerpt = str(card.get("excerpt", ""))[:220]
        source_title = card.get("source_title", "Unknown source")
        source_url = card.get("source_url", "")
        tag = normalize_section_tag(card.get("section_tag", "general"))
        source_type = card.get("source_type", "unknown")
        authority_score = card.get("authority_score", 0)
        lines.append(
            f"- [{tag}] {claim} | {source_title} | {source_type} | authority={authority_score} | {source_url} | excerpt={excerpt}"
        )
    return "\n".join(lines) if lines else "No structured evidence cards were collected."


def _select_section_evidence(section: str, evidence_cards: list[dict], limit: int = 10) -> list[dict]:
    normalized_section = normalize_section_tag(section)
    matching = [card for card in evidence_cards if normalize_section_tag(card.get("section_tag", "")) == normalized_section]
    if len(matching) < limit:
        supplemental = [card for card in evidence_cards if card not in matching]
        supplemental.sort(
            key=lambda card: (
                int(card.get("authority_score", 0)),
                float(card.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        matching.extend(supplemental[: max(0, limit - len(matching))])
    return matching[:limit]


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
    required_sections = _dedupe_items(response.required_sections, max(5, settings.max_initial_tasks + 2))

    return {
        "research_plan": safe_tasks,
        "required_sections": required_sections,
        "pending_tasks": safe_tasks,
        "current_batch": [],
        "gaps": [],
        "human_feedback": "",
        "quality_summary": "",
        "outline_sections": [],
        "section_drafts": {},
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
    evidence_cards = _curate_evidence_cards(state.get("evidence_cards", []), _curated_sources(state))
    required_sections = state.get("required_sections", [])

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
    code_gate_passed = (
        len(distinct_sources) >= settings.min_distinct_sources_for_report
        and authoritative_source_count >= settings.min_authoritative_sources_for_report
        and len(evidence_cards) >= settings.min_evidence_cards_for_report
        and not missing_sections
    )

    messages = [
        SystemMessage(
            content=(
                "You evaluate whether research results are complete and identify only material gaps. "
                "Be strict. Do not approve shallow evidence."
            )
        ),
        HumanMessage(
            content=(
                f"User Query: {state['original_query']}\n\n"
                f"Required Report Sections:\n{chr(10).join(f'- {section}' for section in required_sections) or '- None provided'}\n\n"
                f"Distinct Sources Collected: {len(distinct_sources)}\n"
                f"Authoritative Source Count: {authoritative_source_count}\n"
                f"Structured Evidence Cards: {len(evidence_cards)}\n"
                f"Missing Sections by Deterministic Check:\n{chr(10).join(f'- {section}' for section in missing_sections) or '- None'}\n\n"
                f"Collected Findings:\n{_format_findings(findings)}\n\n"
                f"Collected Evidence:\n{_format_evidence_cards(evidence_cards)}\n\n"
                "Decide whether the answer is complete. For broad advice or research requests, do not mark complete if the evidence is shallow, repetitive, or based on too few distinct or authoritative sources. "
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
    router = get_model_router()
    required_sections = state.get("required_sections", [])
    evidence_cards = _curate_evidence_cards(state.get("evidence_cards", []), _curated_sources(state))
    messages = [
        SystemMessage(content="You design comprehensive, publication-grade report outlines. Your outlines produce reports comparable to professional research firms and Gemini Deep Research."),
        HumanMessage(
            content=(
                f"User Query: {state['original_query']}\n\n"
                f"Required Sections:\n{chr(10).join(f'- {section}' for section in required_sections)}\n\n"
                f"Evidence Snapshot:\n{_format_evidence_cards(evidence_cards, limit=20)}\n\n"
                "Create a COMPREHENSIVE report outline with 6-10 sections. The report must feel like a professional research document.\n\n"
                "REQUIRED STRUCTURE:\n"
                "1. Executive Summary / Overview (always first)\n"
                "2-8. Detailed analytical sections covering all major angles of the query\n"
                "   - Include sections for: market analysis, key players/stakeholders, data & statistics, \n"
                "     trends & developments, policy/regulatory landscape, challenges & risks, \n"
                "     comparative analysis, case studies (as applicable to the topic)\n"
                "9. Future Outlook / Projections\n"
                "10. Conclusion & Recommendations (always last)\n\n"
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
    outline_sections = _dedupe_items(response.sections, max(len(required_sections) + 2, 8)) or required_sections
    return {"outline_sections": outline_sections}


async def draft_sections_node(state: OrchestratorState):
    settings = get_settings()
    router = get_model_router()
    evidence_cards = _curate_evidence_cards(state.get("evidence_cards", []), _curated_sources(state))
    findings = state.get("findings", [])
    section_drafts: dict[str, str] = {}

    for section in state.get("outline_sections", []):
        selected_cards = _select_section_evidence(section, evidence_cards, limit=18)
        evidence_text = _format_evidence_cards(selected_cards, limit=18)
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
                    f"Section Title: {section}\n\n"
                    f"Relevant Evidence:\n{evidence_text}\n\n"
                    f"Supporting Findings:\n{_format_findings(findings, limit=10)}\n\n"
                    "Write a DETAILED Markdown section (300-500 words minimum). Requirements:\n\n"
                    "FORMAT & STRUCTURE:\n"
                    "- Start with a brief contextual intro paragraph for this section\n"
                    "- Use ### subheadings to organize complex information\n"
                    "- Include Markdown TABLES where comparing data (players, pricing, metrics, timelines)\n"
                    "- Use bullet lists for key takeaways or feature comparisons\n"
                    "- End with a brief analytical insight or implication\n\n"
                    "CONTENT QUALITY:\n"
                    "- Include ALL specific numbers, dates, company names, and statistics from the evidence\n"
                    "- Cite sources inline like [Source Name] when referencing specific data points\n"
                    "- Analyze trends and relationships, don't just list facts\n"
                    "- Compare and contrast when multiple data points exist\n"
                    "- Note data gaps or uncertainty explicitly rather than making assumptions\n"
                    "- NO generic filler phrases like 'significant growth' or 'various factors' — be specific"
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


async def final_edit_node(state: OrchestratorState):
    router = get_model_router()
    findings = state.get("findings", [])
    sources = _curated_sources(state)
    evidence_cards = _curate_evidence_cards(state.get("evidence_cards", []), sources)
    section_drafts = state.get("section_drafts", {})
    draft_text = "\n\n".join(f"## {title}\n{content}" for title, content in section_drafts.items())
    reference_section = _build_references_section(evidence_cards)

    messages = [
        SystemMessage(
            content=(
                "You are a senior research editor producing publication-grade reports. "
                "Your output should match the quality of Gemini Deep Research, McKinsey reports, "
                "and professional market research — comprehensive, well-structured, and data-rich. "
                "Write the FULL final report, not a summary."
            )
        ),
        HumanMessage(
            content=(
                f"User Query: {state['original_query']}\n\n"
                f"Quality Summary: {state.get('quality_summary', 'No quality summary available.')}\n\n"
                f"Evidence Snapshot:\n{_format_evidence_cards(evidence_cards, limit=30)}\n\n"
                f"Section Drafts:\n{draft_text}\n\n"
                f"Known Sources:\n{chr(10).join(f'- {source}' for source in sources[:40])}\n\n"
                "WRITE A COMPREHENSIVE FINAL REPORT following these rules:\n\n"
                "STRUCTURE (mandatory):\n"
                "# [Descriptive Report Title]\n"
                "## Executive Summary (3-4 key findings as bullets + brief overview paragraph)\n"
                "## [Section 2-8: Detailed analytical sections from the drafts]\n"
                "## Future Outlook & Projections\n"
                "## Conclusion & Recommendations\n\n"
                "FORMATTING REQUIREMENTS:\n"
                "- Use proper Markdown: # for title, ## for sections, ### for subsections\n"
                "- Include AT LEAST 2-3 Markdown TABLES comparing key data (players, pricing, market shares, metrics, timelines)\n"
                "- Use bullet lists for key takeaways and feature comparisons\n"
                "- Bold **key statistics** and important figures\n"
                "- Include inline source citations like [Source Name]\n\n"
                "CONTENT REQUIREMENTS:\n"
                "- Minimum 3000 words — this is a DETAILED report, not a summary\n"
                "- Integrate ALL specific numbers, dates, and company names from the evidence\n"
                "- Provide analytical insights and trend analysis, not just fact listing\n"
                "- Compare and contrast data points across sources\n"
                "- Note data gaps explicitly\n"
                "- End with actionable, specific recommendations\n"
                "- Do NOT add a references section; it will be appended automatically\n"
                "- Do NOT invent data not present in the evidence — cite only what is supplied"
            )
        ),
    ]
    response = await router.generate_text(
        task_type="synthesis",
        messages=messages,
        budget=RequestBudget(
            max_input_chars=get_settings_instance().synthesis_input_char_budget,
            max_output_tokens=get_settings_instance().final_report_output_tokens,
        ),
        trace_id=state["thread_id"],
    )
    return {"final_report": _replace_references_section(response.content, reference_section)}


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
    if state.get("pending_tasks") and state.get("evaluation_rounds", 0) <= settings.max_gap_rounds:
        return "dispatch_tasks_node"
    return "build_outline_node"


def create_lead_orchestrator(checkpointer=None):
    builder = StateGraph(OrchestratorState)

    builder.add_node("decompose_node", decompose_node)
    builder.add_node("plan_review_node", plan_review_node)
    builder.add_node("dispatch_tasks_node", dispatch_tasks_node)
    builder.add_node("sub_agent", create_sub_agent_graph())
    builder.add_node("post_batch_node", post_batch_node)
    builder.add_node("evaluate_node", evaluate_node)
    builder.add_node("build_outline_node", build_outline_node)
    builder.add_node("draft_sections_node", draft_sections_node)
    builder.add_node("final_edit_node", final_edit_node)

    builder.add_edge(START, "decompose_node")
    builder.add_edge("decompose_node", "plan_review_node")
    builder.add_conditional_edges("plan_review_node", route_human_approval)
    builder.add_conditional_edges("dispatch_tasks_node", route_batch_dispatch)
    builder.add_edge("sub_agent", "post_batch_node")
    builder.add_conditional_edges("post_batch_node", route_after_batch)
    builder.add_conditional_edges("evaluate_node", route_evaluation)
    builder.add_edge("build_outline_node", "draft_sections_node")
    builder.add_edge("draft_sections_node", "final_edit_node")
    builder.add_edge("final_edit_node", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["plan_review_node"],
    )
