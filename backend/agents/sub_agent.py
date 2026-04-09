"""
Sub-Agent — a bounded research worker that uses ModelRouter for tool-calling.
"""

import asyncio
import json
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

AUTHORITATIVE_DOMAINS = (
    ".gov",
    ".edu",
    "aha.org",
    "acc.org",
    "escardio.org",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "mayoclinic.org",
    "clevelandclinic.org",
)


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
- Start with SearchTool, then use ScrapeTool on the most promising results.
- Prefer authoritative sources such as medical societies, journals, government sites, hospitals, and major non-profit organizations.
- Avoid revisiting the same source unless the previous result was unusable.
- Do not stop after only one or two facts. Gather multiple scraped sources and enough evidence to support a rich report section.
- Extract only verifiable facts with source URLs and short supporting excerpts.
- Tag evidence with concise section labels such as diet_pattern, sodium, saturated_fat, fiber, exercise, cardiac_rehab, smoking_alcohol, sleep_stress, or follow_up_cautions when relevant.
- When you truly have enough material, call SubmitFinalFindings with findings, evidence_cards, and coverage_tags.
- Use the provided tool-calling interface. Do not write XML tags, <function=...> wrappers, or raw JSON manually.
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


def infer_source_type(url: str) -> str:
    split = urlsplit(url)
    hostname = split.netloc.lower()
    path = split.path.lower()
    if not hostname:
        return "unknown"
    if hostname.endswith(".gov"):
        return "government"
    if hostname.endswith(".edu"):
        return "academic"
    if "pubmed" in hostname or "ncbi" in hostname or "nejm" in hostname or "thelancet" in hostname or "jamanetwork" in hostname:
        return "journal"
    if any(token in hostname for token in ("clinic", "hospital", "healthsystem", "medicine")):
        return "hospital"
    if any(token in hostname for token in ("heart.org", "acc.org", "escardio.org", "who.int", "nih.gov")):
        return "guideline"
    if any(token in path for token in ("guideline", "guidelines", "statement", "scientific-statement")):
        return "guideline"
    if any(token in hostname for token in ("news", "cnn", "forbes", "reuters")):
        return "news"
    if any(token in hostname for token in ("healthline", "webmd", "verywellhealth")):
        return "commercial_health"
    return "nonprofit" if hostname.endswith(".org") else "commercial"


def compute_authority_score(url: str, source_type: str | None = None) -> int:
    hostname = urlsplit(url).netloc.lower()
    source_type = source_type or infer_source_type(url)
    if any(domain in hostname for domain in AUTHORITATIVE_DOMAINS):
        return 10
    if source_type in {"guideline", "government", "journal"}:
        return 9
    if source_type in {"academic", "hospital", "nonprofit"}:
        return 8
    if source_type == "news":
        return 5
    if source_type == "commercial_health":
        return 4
    if source_type == "commercial":
        return 3
    return 2


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
            lines.append(f"- {title} | {url} | {snippet[:160]}")
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


async def reason_node(state: SubAgentState) -> dict:
    settings = _settings()
    router = get_model_router()

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
            result = "Findings successfully recorded."
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
    last_message = state["messages"][-1]
    status = "done"
    if state.get("iterations", 0) >= get_settings().max_sub_agent_iterations:
        status = "max_iterations"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "SubmitFinalFindings":
                try:
                    validated = SubmitFinalFindings(**tool_call["args"])
                    tool_message = ToolMessage(content="Findings submitted gracefully.", tool_call_id=tool_call["id"])
                    evidence_cards = [_evidence_to_dict(card) for card in validated.evidence_cards]
                    coverage_tags = _dedupe_strings(
                        [normalize_section_tag(tag) for tag in validated.coverage_tags]
                        + [normalize_section_tag(card["section_tag"]) for card in evidence_cards if card.get("section_tag")]
                    )
                    scraped_sources = _dedupe_strings(
                        list(state.get("sources", []))
                        + [card["source_url"] for card in evidence_cards if card.get("source_url")]
                        + [finding.source_url for finding in validated.findings if finding.source_url]
                    )
                    discovered_sources = _dedupe_strings(
                        list(state.get("discovered_sources", []))
                        + scraped_sources
                    )
                    return {
                        "messages": [tool_message],
                        "findings": [_finding_to_dict(finding) for finding in validated.findings],
                        "evidence_cards": evidence_cards,
                        "coverage_tags": coverage_tags,
                        "sources": scraped_sources,
                        "discovered_sources": discovered_sources,
                        "completed_tasks": [state["task"]],
                        "status": status,
                    }
                except ValidationError:
                    break

    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    fallback_source = state.get("sources", ["parsed_from_conversation"])[0] if state.get("sources") else "parsed_from_conversation"
    findings = [_finding_to_dict(Finding(fact=str(content)[:500], source_url=fallback_source, confidence=0.5))]
    source_type = infer_source_type(fallback_source)
    evidence_cards = [
        _evidence_to_dict(
            EvidenceCard(
                claim=str(content)[:300],
                source_url=fallback_source,
                source_title=urlsplit(fallback_source).netloc or "Fallback source",
                excerpt=state.get("working_summary", "")[:500] or str(content)[:500],
                section_tag=normalize_section_tag(state["task"]),
                source_type=source_type,
                authority_score=compute_authority_score(fallback_source, source_type),
                confidence=0.5,
            )
        )
    ]
    return {
        "findings": findings,
        "evidence_cards": evidence_cards,
        "coverage_tags": [normalize_section_tag(state["task"])],
        "sources": _dedupe_strings(list(state.get("sources", [])) + [fallback_source]),
        "discovered_sources": _dedupe_strings(list(state.get("discovered_sources", [])) + [fallback_source]),
        "completed_tasks": [state["task"]],
        "status": status,
    }


def should_continue(state: SubAgentState) -> str:
    if state.get("iterations", 0) >= get_settings().max_sub_agent_iterations:
        return "finalize"

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "SubmitFinalFindings":
                try:
                    SubmitFinalFindings(**tool_call["args"])
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
