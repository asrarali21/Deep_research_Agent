"""
Sub-Agent — a bounded research worker that uses ModelRouter for tool-calling.
"""

import asyncio
import json
import logging
from typing import Annotated, TypedDict
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ValidationError

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

logger = logging.getLogger("sub_agent")

# Debug file logger for tracing evidence pipeline
_debug_handler = logging.FileHandler("/tmp/deep_research_debug.log")
_debug_handler.setLevel(logging.DEBUG)
_debug_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
logger.addHandler(_debug_handler)
logger.setLevel(logging.DEBUG)
from tools.firecrawl_tool import scrape, search

load_dotenv()


SECTION_TAG_ALIASES: dict[str, tuple[str, ...]] = {
    "diet_pattern": ("diet", "mediterranean", "dash", "meal pattern", "meal plan", "nutrition pattern"),
    "sodium": ("sodium", "salt"),
    "saturated_fat": ("saturated fat", "fat"),
    "fiber": ("fiber", "fibre", "whole grain", "whole grains"),
    "exercise": ("exercise", "physical activity", "fitness", "walking", "activity"),
    "cardiac_rehab": ("cardiac rehab", "cardiac rehabilitation", "rehab", "rehabilitation"),
    "smoking_alcohol": ("smoking", "alcohol", "tobacco"),
    "sleep_stress": ("sleep", "stress", "mental health"),
    "follow_up_cautions": ("follow up", "follow-up", "medication", "drug", "caution", "warning", "monitoring"),
}

class SubAgentState(TypedDict):
    trace_id: str
    task: str
    messages: Annotated[list, add_messages]
    working_summary: str
    findings: list[dict]
    evidence_cards: list[dict]
    sources: list[str]
    discovered_sources: list[str]
    seen_source_urls: list[str]
    coverage_tags: list[str]
    completed_tasks: list[str]
    iterations: int
    status: str


SYSTEM_PROMPT = """You are a quality-first research agent. Your job is to build a strong evidence pack for one specific research task.

WORKFLOW:
- Start with SearchTool (1-2 focused searches), then ScrapeTool on the 2-3 most promising results.
- After scraping, extract key facts and call SubmitFinalFindings.

EFFICIENCY (critical — API budget is limited):
- Be decisive: 1-2 searches + 2-3 scrapes + submit = done. Do NOT over-research.
- If you already have 3+ solid evidence cards with specific facts, numbers, or quotes — SUBMIT immediately.
- Never run more than 3 search queries total. Refine your query instead of repeating.

RELEVANCE (critical — off-topic evidence is rejected):
- Every evidence card MUST directly answer the research task. Do NOT submit general industry reports, unrelated market data, or tangential commercial content.
- If search results are irrelevant to the task, REFINE your search query to be more specific. Do not scrape irrelevant pages.
- Before submitting, mentally verify: "Does this evidence directly help answer the research task?" If not, discard it.
- Off-topic evidence cards will be automatically filtered out and won't count toward the minimum requirements.

QUALITY:
- Prefer authoritative sources: government sites, journals, official reports, industry databases.
- For market/industry tasks: government notifications, company announcements, exchange filings, reputable analysis with exact figures.
- Extract only verifiable facts with source URLs and short supporting excerpts.
- Prefer exact numbers, dates, named companies, pricing, regulations over generic background.
- Tag evidence with concise section labels (e.g., market_share, pricing, policy, technology).
- Avoid revisiting the same source unless the previous result was unusable.

TOOLS:
- Use the provided tool-calling interface. Do not write XML tags, <function=...> wrappers, or raw JSON manually.
- When you have enough material, call SubmitFinalFindings with findings, evidence_cards, and coverage_tags.
"""


class Finding(BaseModel):
    fact: str = Field(description="A clear factual statement")
    source_url: str = Field(description="The URL supporting the fact")
    confidence: float = Field(description="A confidence score between 0.0 and 1.0")


class EvidenceCard(BaseModel):
    claim: str = Field(description="A concrete evidence-backed claim or recommendation")
    source_url: str = Field(description="The URL supporting the claim")
    source_title: str = Field(description="A short source title")
    excerpt: str = Field(description="A short supporting excerpt or note from the source")
    section_tag: str = Field(description="A concise section tag such as sodium, cardiac_rehab, or exercise")
    source_type: str = Field(description="The source type such as guideline, government, journal, hospital, nonprofit, news, or commercial")
    authority_score: int = Field(description="A rough authority score between 1 and 10")
    confidence: float = Field(description="A confidence score between 0.0 and 1.0")


class SubmitFinalFindings(BaseModel):
    findings: list[Finding] = Field(description="Verified findings with sources")
    evidence_cards: list[EvidenceCard] = Field(description="Structured evidence cards with excerpts, section tags, and source quality")
    coverage_tags: list[str] = Field(description="The main section tags covered by this research task")
    summary: str = Field(description="A 2-4 sentence summary of what was learned")


class SearchTool(BaseModel):
    query: str = Field(description="The search query string")


class ScrapeTool(BaseModel):
    url: str = Field(description="The full URL to scrape")


TOOL_MAP = {
    "SearchTool": search,
    "ScrapeTool": scrape,
    "SubmitFinalFindings": None,
}


def normalize_section_tag(value: str) -> str:
    cleaned = " ".join(value.replace("_", " ").replace("-", " ").lower().split())
    if not cleaned:
        return "general"
    for canonical, aliases in SECTION_TAG_ALIASES.items():
        if cleaned == canonical.replace("_", " "):
            return canonical
        if any(alias in cleaned for alias in aliases):
            return canonical
    return cleaned.replace(" ", "_")
def _dedupe_strings(items: list[str]) -> list[str]:
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
    return unique_items


def _finding_to_dict(finding: Finding | dict) -> dict:
    if isinstance(finding, Finding):
        return finding.model_dump()
    return dict(finding)


def _evidence_to_dict(card: EvidenceCard | dict) -> dict:
    if isinstance(card, EvidenceCard):
        return card.model_dump()
    return dict(card)


def _settings():
    return get_settings_instance()


def _truncate_tool_result(result):
    settings = get_settings()
    char_limit = settings.tool_result_char_limit

    if isinstance(result, str) and len(result) > char_limit:
        return result[:char_limit] + "\n\n... [Content Truncated] ..."

    if isinstance(result, dict):
        trimmed = dict(result)
        if "content" in trimmed and isinstance(trimmed["content"], str) and len(trimmed["content"]) > char_limit:
            trimmed["content"] = trimmed["content"][:char_limit] + "\n\n... [Content Truncated] ..."
        if "snippet" in trimmed and isinstance(trimmed["snippet"], str) and len(trimmed["snippet"]) > 1000:
            trimmed["snippet"] = trimmed["snippet"][:1000] + "..."
        return trimmed

    if isinstance(result, list):
        trimmed_items = []
        for item in result[: get_settings().search_result_limit]:
            if isinstance(item, dict):
                trimmed_item = dict(item)
                if "snippet" in trimmed_item and isinstance(trimmed_item["snippet"], str) and len(trimmed_item["snippet"]) > 500:
                    trimmed_item["snippet"] = trimmed_item["snippet"][:500] + "..."
                trimmed_items.append(trimmed_item)
            else:
                trimmed_items.append(item)
        return trimmed_items

    return result


def _extract_summary_fragment(tool_name: str, result) -> str:
    if tool_name == "SearchTool" and isinstance(result, list):
        lines = ["Search results:"]
        for item in result[: min(len(result), 5)]:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            source_type = item.get("source_type", "unknown")
            authority = item.get("authority_score", 0)
            relevance = round(float(item.get("relevance_score", 0.0)), 1)
            lines.append(f"- {title} | {url} | {source_type} | authority={authority} | score={relevance} | {snippet[:140]}")
        return "\n".join(lines)

    if tool_name == "ScrapeTool" and isinstance(result, dict):
        content = result.get("content", "")
        excerpt = content[:600].replace("\n\n", "\n")
        return f"Scraped {result.get('url', '')}:\n{excerpt}"

    return str(result)[:600]


def _merge_working_summary(existing: str, tool_name: str, result) -> str:
    settings = get_settings()
    new_fragment = _extract_summary_fragment(tool_name, result).strip()
    if not new_fragment:
        return existing
    combined = f"{existing}\n\n{new_fragment}".strip() if existing else new_fragment
    if len(combined) > settings.working_summary_char_limit:
        combined = combined[-settings.working_summary_char_limit :]
    return combined


def _validated_evidence_cards(raw_cards: list[dict], scraped_sources: list[str]) -> list[dict]:
    allowed_sources = set(scraped_sources)
    valid_cards: list[dict] = []
    seen = set()
    for card in raw_cards:
        source_url = str(card.get("source_url", "")).strip()
        claim = str(card.get("claim", "")).strip()
        excerpt = str(card.get("excerpt", "")).strip()
        if not source_url or not claim or not excerpt:
            continue
        if source_url not in allowed_sources:
            continue
        if not is_reference_usable(source_url) or is_blocked_reference_url(source_url):
            continue
        source_type = card.get("source_type") or infer_source_type(source_url)
        authority_score = int(card.get("authority_score", compute_authority_score(source_url, source_type)))
        cleaned = {
            **card,
            "source_url": source_url,
            "claim": claim,
            "excerpt": excerpt[:450],
            "section_tag": normalize_section_tag(card.get("section_tag", "")),
            "source_type": source_type,
            "authority_score": authority_score,
            "source_title": str(card.get("source_title", "")).strip() or urlsplit(source_url).netloc,
        }
        key = (cleaned["section_tag"], source_url, cleaned["claim"].lower())
        if key in seen:
            continue
        seen.add(key)
        valid_cards.append(cleaned)
    return valid_cards


def _validated_findings(raw_findings: list[Finding], scraped_sources: list[str]) -> list[dict]:
    allowed_sources = set(scraped_sources)
    validated: list[dict] = []
    seen = set()
    for finding in raw_findings:
        source_url = finding.source_url.strip()
        if source_url not in allowed_sources or not is_reference_usable(source_url):
            continue
        key = (finding.fact.strip().lower(), source_url)
        if key in seen:
            continue
        seen.add(key)
        validated.append(_finding_to_dict(finding))
    return validated


def _empty_submission_result(state: SubAgentState, status: str) -> dict:
    return {
        "findings": [],
        "evidence_cards": [],
        "coverage_tags": [],
        "sources": _dedupe_strings(list(state.get("sources", []))),
        "discovered_sources": _dedupe_strings(list(state.get("discovered_sources", []))),
        "completed_tasks": [state["task"]],
        "status": status,
    }


def _task_tokens(task: str) -> set[str]:
    """Extract meaningful tokens from a task description for relevance matching."""
    from services.source_quality import query_tokens
    return set(query_tokens(task))


def _is_evidence_relevant_to_task(task: str, card: dict, threshold: float = 0.15) -> bool:
    """Check if an evidence card is topically relevant to the research task.
    
    Returns True if at least `threshold` fraction of the task's tokens appear
    in the card's claim + excerpt + title via fuzzy matching (stemming + substring).
    This catches obviously off-topic evidence like spa market reports showing up
    in an AGI research task, while allowing for natural language inflections.
    """
    from services.source_quality import query_tokens, fuzzy_token_overlap
    task_token_set = set(query_tokens(task))
    if not task_token_set:
        return True  # No meaningful tokens in task, skip filter
    card_text = " ".join([
        str(card.get("claim", "")),
        str(card.get("excerpt", "")),
        str(card.get("source_title", "")),
    ])
    hits = fuzzy_token_overlap(task_token_set, card_text)
    return (hits / len(task_token_set)) >= threshold


def assess_submission_quality(state: SubAgentState, submitted: SubmitFinalFindings) -> tuple[list[str], list[dict], list[dict], list[str]]:
    settings = get_settings()
    scraped_sources = _dedupe_strings(list(state.get("sources", [])))
    valid_evidence_cards = _validated_evidence_cards(
        [_evidence_to_dict(card) for card in submitted.evidence_cards],
        scraped_sources,
    )
    valid_findings = _validated_findings(submitted.findings, scraped_sources)

    # --- Relevance filter: drop evidence cards that are off-topic to the task ---
    task_text = state.get("task", "")
    total_before_relevance = len(valid_evidence_cards)
    valid_evidence_cards = [
        card for card in valid_evidence_cards
        if _is_evidence_relevant_to_task(task_text, card)
    ]
    off_topic_count = total_before_relevance - len(valid_evidence_cards)

    authoritative_sources = {
        card["source_url"]
        for card in valid_evidence_cards
        if int(card.get("authority_score", 0)) >= 8 and not is_low_value_reference_url(card["source_url"])
    }
    coverage_tags = _dedupe_strings(
        [normalize_section_tag(tag) for tag in submitted.coverage_tags]
        + [normalize_section_tag(card.get("section_tag", "")) for card in valid_evidence_cards]
    )

    issues: list[str] = []

    # Relevance quality gate
    if total_before_relevance > 0 and off_topic_count > total_before_relevance * 0.5:
        issues.append(
            f"{off_topic_count} of {total_before_relevance} evidence cards were off-topic and filtered out. "
            "Refine your search queries to be more specific to the research task. "
            "Do not submit evidence from unrelated industries or topics."
        )

    if len(scraped_sources) < settings.min_scraped_sources_per_task:
        issues.append(f"Need at least {settings.min_scraped_sources_per_task} scraped sources before finalizing.")
    if len(valid_evidence_cards) < settings.min_evidence_cards_per_task:
        issues.append(
            f"Need at least {settings.min_evidence_cards_per_task} evidence cards backed by scraped sources before finalizing."
        )
    if len(authoritative_sources) < settings.min_authoritative_sources_per_task:
        issues.append(
            f"Need at least {settings.min_authoritative_sources_per_task} authoritative scraped source before finalizing."
        )
    if not valid_findings:
        issues.append("Need at least one validated finding backed by a scraped source.")

    if any(looks_like_homepage(url) for url in scraped_sources) and len(scraped_sources) <= settings.min_scraped_sources_per_task:
        issues.append("Do not rely mainly on homepage URLs. Scrape article, report, policy, or guideline pages.")

    return issues, valid_findings, valid_evidence_cards, coverage_tags


async def reason_node(state: SubAgentState) -> dict:
    settings = _settings()
    router = get_model_router()

    logger.debug(
        "reason_node: task=%s, iterations=%d, sources=%d, evidence_cards=%d",
        state["task"][:80], state.get("iterations", 0),
        len(state.get("sources", [])), len(state.get("evidence_cards", [])),
    )
    recent_messages = state["messages"][-settings.recent_message_count :]
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=f"Research task:\n{state['task']}")]
    messages.append(
        SystemMessage(
            content=(
                "Current progress:\n"
                f"- Scraped sources: {len(state.get('sources', []))}\n"
                f"- Candidate sources seen: {len(state.get('discovered_sources', []))}\n"
                f"- Evidence cards captured: {len(state.get('evidence_cards', []))}\n"
                "Do not finalize until the evidence pack is substantively useful for a detailed report section."
            )
        )
    )
    if state.get("working_summary"):
        messages.append(
            SystemMessage(
                content=(
                    "Working memory summary from prior tool results:\n"
                    f"{state['working_summary']}"
                )
            )
        )
    messages.extend(recent_messages)

    response = await router.generate_tool_calling(
        task_type="worker_tool_calling",
        messages=messages,
        tools=[SearchTool, ScrapeTool, SubmitFinalFindings],
        budget=RequestBudget(max_input_chars=settings.worker_input_char_budget, max_output_tokens=800),
        trace_id=state["trace_id"],
    )

    tool_names = [tc["name"] for tc in getattr(response, "tool_calls", []) or []]
    logger.debug("reason_node: LLM responded with tool_calls=%s", tool_names)

    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1,
    }


def act_node(state: SubAgentState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []
    new_sources = list(state.get("sources", []))
    discovered_sources = list(state.get("discovered_sources", []))
    seen_source_urls = list(state.get("seen_source_urls", []))
    working_summary = state.get("working_summary", "")

    logger.debug("act_node: processing %d tool_calls", len(getattr(last_message, "tool_calls", []) or []))

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_fn = TOOL_MAP.get(tool_name)

        if tool_name == "ScrapeTool" and tool_args.get("url") in seen_source_urls:
            result = {
                "url": tool_args.get("url", ""),
                "title": "",
                "content": "Source already visited during this job. Reuse the previous evidence or choose another source.",
                "success": True,
            }
        elif tool_fn:
            try:
                result = tool_fn(**tool_args)
                result = _truncate_tool_result(result)
            except ValidationError as error:
                result = f"Pydantic Validation Error: {error}"
            except Exception as error:
                result = f"Execution Error: {error}"
        elif tool_name == "SubmitFinalFindings":
            try:
                submitted = SubmitFinalFindings(**tool_args)
                issues, valid_findings, valid_evidence_cards, coverage_tags = assess_submission_quality(state, submitted)
                logger.debug(
                    "act_node: SubmitFinalFindings — raw_cards=%d, valid_cards=%d, valid_findings=%d, issues=%s",
                    len(submitted.evidence_cards), len(valid_evidence_cards), len(valid_findings), issues,
                )
                if issues and state.get("iterations", 0) < get_settings().max_sub_agent_iterations:
                    result = {
                        "accepted": False,
                        "issues": issues,
                        "validated_findings": len(valid_findings),
                        "validated_evidence_cards": len(valid_evidence_cards),
                        "coverage_tags": coverage_tags,
                    }
                else:
                    result = {
                        "accepted": True,
                        "validated_findings": len(valid_findings),
                        "validated_evidence_cards": len(valid_evidence_cards),
                    }
            except ValidationError as error:
                result = f"Pydantic Validation Error: {error}"
        else:
            result = f"Error: Unknown tool '{tool_name}'"

        if tool_name == "SearchTool" and isinstance(result, list):
            for item in result:
                url = item.get("url", "")
                if url and url not in discovered_sources:
                    discovered_sources.append(url)

        if tool_name == "ScrapeTool" and isinstance(result, dict):
            url = result.get("url", "")
            if url and url not in discovered_sources:
                discovered_sources.append(url)
            if url and url not in seen_source_urls:
                seen_source_urls.append(url)
                new_sources.append(url)

        if isinstance(result, (dict, list)):
            working_summary = _merge_working_summary(working_summary, tool_name, result)
            result_str = json.dumps(result, indent=2)
        else:
            result_str = str(result)

        tool_messages.append(
            ToolMessage(
                content=result_str,
                tool_call_id=tool_call["id"],
            )
        )

    return {
        "messages": tool_messages,
        "sources": new_sources,
        "discovered_sources": discovered_sources,
        "seen_source_urls": seen_source_urls,
        "working_summary": working_summary,
    }


def finalize_node(state: SubAgentState) -> dict:
    status = "done"
    if state.get("iterations", 0) >= get_settings().max_sub_agent_iterations:
        status = "max_iterations"

    logger.debug(
        "finalize_node: status=%s, iterations=%d, total_messages=%d, sources=%d",
        status, state.get("iterations", 0), len(state["messages"]), len(state.get("sources", [])),
    )

    # Count messages with tool_calls
    messages_with_submit = 0
    for msg in state["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "SubmitFinalFindings":
                    messages_with_submit += 1
    logger.debug("finalize_node: found %d SubmitFinalFindings calls in message history", messages_with_submit)

    # Scan ALL messages for the most recent SubmitFinalFindings call.
    # The last message may be a ToolMessage (from act_node), not the AIMessage
    # that contained the tool_calls. We need to walk backwards to find it.
    for message in reversed(state["messages"]):
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            continue
        for tool_call in message.tool_calls:
            if tool_call["name"] != "SubmitFinalFindings":
                continue
            try:
                validated = SubmitFinalFindings(**tool_call["args"])
                issues, valid_findings, valid_evidence_cards, coverage_tags = assess_submission_quality(state, validated)
                if issues and status != "max_iterations":
                    # Still has issues and not at max iterations — keep trying
                    continue
                tool_message = ToolMessage(content="Findings submitted gracefully.", tool_call_id=tool_call["id"])
                scraped_sources = _dedupe_strings(
                    list(state.get("sources", []))
                    + [card["source_url"] for card in valid_evidence_cards if card.get("source_url")]
                    + [finding["source_url"] for finding in valid_findings if finding.get("source_url")]
                )
                discovered_sources = _dedupe_strings(
                    list(state.get("discovered_sources", []))
                    + scraped_sources
                )
                return {
                    "messages": [tool_message],
                    "findings": valid_findings,
                    "evidence_cards": valid_evidence_cards,
                    "coverage_tags": coverage_tags,
                    "sources": scraped_sources,
                    "discovered_sources": discovered_sources,
                    "completed_tasks": [state["task"]],
                    "status": status,
                }
            except ValidationError as e:
                logger.debug("finalize_node: ValidationError for SubmitFinalFindings: %s", e)
                continue

    logger.warning(
        "finalize_node: NO valid SubmitFinalFindings found! Returning empty. status=%s, task=%s",
        status, state["task"][:80],
    )
    if status == "done":
        status = "insufficient_evidence"
    return _empty_submission_result(state, status)


def should_continue(state: SubAgentState) -> str:
    if state.get("iterations", 0) >= get_settings().max_sub_agent_iterations:
        return "finalize"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "SubmitFinalFindings":
                try:
                    submitted = SubmitFinalFindings(**tool_call["args"])
                    issues, _, _, _ = assess_submission_quality(state, submitted)
                    if issues:
                        return "act"
                    return "finalize"
                except ValidationError:
                    return "act"
        return "act"
    return "finalize"


def create_sub_agent_graph():
    graph = StateGraph(SubAgentState)
    graph.add_node("reason", reason_node)
    graph.add_node("act", act_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "reason")
    graph.add_conditional_edges("reason", should_continue)
    graph.add_edge("act", "reason")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_sub_agent(task: str, thread_id: str = "local_test") -> dict:
    graph = create_sub_agent_graph()
    initial_state = {
        "trace_id": thread_id,
        "task": task,
        "messages": [HumanMessage(content=f"Research the following topic thoroughly:\n\n{task}")],
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
    }

    final_state = asyncio.run(graph.ainvoke(initial_state))
    return {
        "findings": final_state.get("findings", []),
        "evidence_cards": final_state.get("evidence_cards", []),
        "sources": final_state.get("sources", []),
        "discovered_sources": final_state.get("discovered_sources", []),
        "coverage_tags": final_state.get("coverage_tags", []),
        "iterations": final_state.get("iterations", 0),
        "status": final_state.get("status", "unknown"),
    }
