"""
Sub-Agent — a bounded research worker that uses ModelRouter for tool-calling.
"""

import asyncio
import json
from typing import Annotated, TypedDict

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


class SubAgentState(TypedDict):
    trace_id: str
    task: str
    messages: Annotated[list, add_messages]
    working_summary: str
    findings: list[dict]
    sources: list[str]
    seen_source_urls: list[str]
    iterations: int
    status: str


SYSTEM_PROMPT = """You are a focused research agent. Your job is to research a specific topic using the available search and page-reading tools.
- Start with SearchTool, then use ScrapeTool on promising results.
- Avoid revisiting the same source unless the previous result was unusable.
- Extract only verifiable facts with source URLs.
- When you have enough material, call SubmitFinalFindings.
- Keep tool calls in standard JSON format only.
"""


class Finding(BaseModel):
    fact: str = Field(description="A clear factual statement")
    source_url: str = Field(description="The URL supporting the fact")
    confidence: float = Field(description="A confidence score between 0.0 and 1.0")


class SubmitFinalFindings(BaseModel):
    findings: list[Finding] = Field(description="Verified findings with sources")
    summary: str = Field(description="A 2-3 sentence summary of what was learned")


class SearchTool(BaseModel):
    query: str = Field(description="The search query string")


class ScrapeTool(BaseModel):
    url: str = Field(description="The full URL to scrape")


TOOL_MAP = {
    "SearchTool": search,
    "ScrapeTool": scrape,
    "SubmitFinalFindings": None,
}


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
        for item in result[:5]:
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
        for item in result[:3]:
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
        budget=RequestBudget(max_input_chars=settings.worker_input_char_budget),
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

        if tool_name == "ScrapeTool" and isinstance(result, dict):
            url = result.get("url", "")
            if url and url not in seen_source_urls:
                seen_source_urls.append(url)
                new_sources.append(url)
        elif tool_name == "SearchTool" and isinstance(result, list):
            for item in result:
                url = item.get("url", "")
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
                    return {
                        "messages": [tool_message],
                        "findings": validated.findings,
                        "status": status,
                    }
                except ValidationError:
                    break

    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    fallback_source = state.get("sources", ["parsed_from_conversation"])[0] if state.get("sources") else "parsed_from_conversation"
    findings = [Finding(fact=str(content)[:500], source_url=fallback_source, confidence=0.5)]
    return {
        "findings": findings,
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
        "sources": [],
        "seen_source_urls": [],
        "iterations": 0,
        "status": "running",
    }

    final_state = asyncio.run(graph.ainvoke(initial_state))
    return {
        "findings": final_state.get("findings", []),
        "sources": final_state.get("sources", []),
        "iterations": final_state.get("iterations", 0),
        "status": final_state.get("status", "unknown"),
    }
