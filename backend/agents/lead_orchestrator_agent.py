"""
Lead Orchestrator — batches research work and routes every model call through ModelRouter.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from agents.sub_agent import create_sub_agent_graph
from services.config import get_settings
from services.model_router import RequestBudget
from services.runtime import get_model_router, get_settings_instance


class OrchestratorState(TypedDict):
    thread_id: str
    original_query: str
    research_plan: list[str]
    pending_tasks: list[str]
    current_batch: list[str]
    human_feedback: str
    findings: Annotated[list[dict], operator.add]
    sources: Annotated[list[str], operator.add]
    gaps: list[str]
    evaluation_rounds: int
    final_report: str


class DecomposePlan(BaseModel):
    tasks: list[str] = Field(
        description="Highly specific research tasks that isolated workers can handle independently.",
    )


class EvaluateGaps(BaseModel):
    is_complete: bool = Field(description="True when the collected facts are enough to answer the query.")
    gaps: list[str] = Field(description="Specific missing questions that need another round of research.")


def _dedupe_items(items: list[str], limit: int) -> list[str]:
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
        if len(unique_items) >= limit:
            break
    return unique_items


async def decompose_node(state: OrchestratorState):
    settings = get_settings()
    router = get_model_router()
    query = state["original_query"]
    feedback = state.get("human_feedback", "")

    prompt = (
        "Break the user query into highly specific research tasks that can be run independently. "
        f"Create at most {settings.max_initial_tasks} tasks, ordered by importance. "
        "Favor breadth first, avoid duplicate work, and keep each task self-contained."
    )
    if feedback:
        prompt += f"\nIncorporate this human feedback into the revised plan: {feedback}"

    messages = [
        SystemMessage(content="You are the planning brain for a production research system."),
        HumanMessage(content=f"User Query: {query}\n\n{prompt}"),
    ]
    response = await router.generate_structured(
        task_type="planner",
        schema=DecomposePlan,
        messages=messages,
        budget=RequestBudget(max_input_chars=get_settings_instance().planner_input_char_budget, max_output_tokens=400),
        trace_id=state["thread_id"],
    )
    safe_tasks = _dedupe_items(response.tasks, settings.max_initial_tasks)

    return {
        "research_plan": safe_tasks,
        "pending_tasks": safe_tasks,
        "current_batch": [],
        "gaps": [],
        "human_feedback": "",
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
    distinct_sources = list(
        dict.fromkeys(
            [
                f.get("source_url", "")
                for f in findings
                if isinstance(f, dict) and f.get("source_url")
            ]
        )
    )
    findings_text = "\n".join(
        [
            f"- {f.fact if hasattr(f, 'fact') else f.get('fact', 'Unknown')} "
            f"(Source {f.source_url if hasattr(f, 'source_url') else f.get('source_url', 'Unknown')}; "
            f"Confidence {f.confidence if hasattr(f, 'confidence') else f.get('confidence', 0.5)})"
            for f in findings
        ]
    )
    if not findings_text:
        findings_text = "No facts were found by the workers."

    messages = [
        SystemMessage(content="You evaluate whether research results are complete and identify only material gaps."),
        HumanMessage(
            content=(
                f"User Query: {state['original_query']}\n\n"
                f"Distinct Sources Collected: {len(distinct_sources)}\n\n"
                f"Collected Findings:\n{findings_text}\n\n"
                "Decide whether the answer is complete. For broad advice or research requests, do not mark complete if the evidence is shallow, repetitive, or based on too few distinct sources. "
                "If not complete, provide only the most important missing questions."
            )
        ),
    ]
    response = await router.generate_structured(
        task_type="evaluator",
        schema=EvaluateGaps,
        messages=messages,
        budget=RequestBudget(max_input_chars=get_settings_instance().planner_input_char_budget, max_output_tokens=300),
        trace_id=state["thread_id"],
    )

    next_gap_rounds = state.get("evaluation_rounds", 0)
    pending_gap_tasks: list[str] = []
    if not response.is_complete:
        pending_gap_tasks = _dedupe_items(response.gaps, settings.max_gap_tasks_per_round)
        if pending_gap_tasks:
            next_gap_rounds += 1

    return {
        "gaps": pending_gap_tasks,
        "pending_tasks": [f"Supplemental Gap Research: {gap}" for gap in pending_gap_tasks],
        "evaluation_rounds": next_gap_rounds,
    }


async def synthesize_node(state: OrchestratorState):
    router = get_model_router()
    findings = state.get("findings", [])
    sources = list(
        dict.fromkeys(
            [source for source in state.get("sources", []) if source]
            + [f.get("source_url", "") for f in findings if isinstance(f, dict) and f.get("source_url")]
        )
    )
    findings_text = "\n".join(
        [
            f"- {f.fact if hasattr(f, 'fact') else f.get('fact', 'Unknown')} "
            f"(Source: {f.source_url if hasattr(f, 'source_url') else f.get('source_url', 'Unknown')})"
            for f in findings
        ]
    )

    messages = [
        SystemMessage(content="You write production-grade reports using only verified facts."),
        HumanMessage(
            content=(
                f"User Query: {state['original_query']}\n\n"
                f"Verified Findings:\n{findings_text}\n\n"
                f"Known Sources:\n{chr(10).join(f'- {source}' for source in sources)}\n\n"
                "Write a thorough Markdown report, not a brief summary. "
                "Include: a short introduction, key evidence, practical recommendations, cautions or uncertainty where relevant, and a conclusion. "
                "Use only the supplied findings. End with a numbered reference list that includes every materially used source URL."
            )
        ),
    ]
    response = await router.generate_text(
        task_type="synthesis",
        messages=messages,
        budget=RequestBudget(max_input_chars=get_settings_instance().synthesis_input_char_budget, max_output_tokens=2200),
        trace_id=state["thread_id"],
    )
    return {"final_report": response.content}


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
                    "sources": [],
                    "seen_source_urls": [],
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
    return "synthesize_node"


def create_lead_orchestrator(checkpointer=None):
    builder = StateGraph(OrchestratorState)

    builder.add_node("decompose_node", decompose_node)
    builder.add_node("plan_review_node", plan_review_node)
    builder.add_node("dispatch_tasks_node", dispatch_tasks_node)
    builder.add_node("sub_agent", create_sub_agent_graph())
    builder.add_node("post_batch_node", post_batch_node)
    builder.add_node("evaluate_node", evaluate_node)
    builder.add_node("synthesize_node", synthesize_node)

    builder.add_edge(START, "decompose_node")
    builder.add_edge("decompose_node", "plan_review_node")
    builder.add_conditional_edges("plan_review_node", route_human_approval)
    builder.add_conditional_edges("dispatch_tasks_node", route_batch_dispatch)
    builder.add_edge("sub_agent", "post_batch_node")
    builder.add_conditional_edges("post_batch_node", route_after_batch)
    builder.add_conditional_edges("evaluate_node", route_evaluation)
    builder.add_edge("synthesize_node", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["plan_review_node"],
    )
