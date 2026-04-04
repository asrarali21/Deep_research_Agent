"""
Lead Orchestrator — The Manager of the Deep Research System (LangGraph).

This is Step 3 of the build.

HOW IT WORKS (Map-Reduce & Interrupts):
───────────────────────────────────────
  1. USER QUERY: Receives a complex query.
  2. DECOMPOSE: Breaks query into dynamic `N` parallel research tasks.
  3. INTERRUPT: Pauses graph. User reviews the tasks in terminal.
     - If user edits: loops back to step 2 to redraw plan.
     - If user approves: proceed.
  4. SPAWN (Map): Uses LangGraph `.Send()` to boot up `N` Sub-Agents simultaneously.
  5. EVALUATE (Reduce): Waited for all Sub-Agents to finish. Reads all their facts. 
     Are there gaps in answering the query?
     - If gaps exist (and under 3 loops): Send missing questions to NEW Sub-Agents.
     - If perfect: proceed.
  6. SYNTHESIZE: Write the final comprehensive Markdown report.
"""

import os
import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Import the exact Sub-Agent logic we built in Step 2!
from agents.sub_agent import create_sub_agent_graph, Finding

load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1: ORCHESTRATOR LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# We use Gemini 1.5 Pro / 2.0 Flash for the Orchestrator. 
# While the Sub-Agents use fast Groq models to browse the web, 
# the Orchestrator needs complex reasoning to plan, map, and evaluate.

def create_orchestrator_llm():
    return ChatOpenAI(
        model="orchestrator-model",  # Maps to liteLLM config (Gemini -> Groq fallback)
        api_key="litellm",
        base_url="http://0.0.0.0:4000",
        temperature=0.2  # Low temperature for highly logical planning
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2: THE STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The global memory tracking the entire project lifecycle.
# Notice `operator.add` on `findings`. When 5 parallel sub-agents finish 
# and return their `findings` arrays, LangGraph auto-adds them into this master list!

class OrchestratorState(TypedDict):
    original_query: str
    research_plan: list[str]
    human_feedback: str          # Any edits the user suggests during the interrupt
    
    # State Reducers: Auto-accumulate data from parallel workers
    findings: Annotated[list[Finding], operator.add] 
    sources: Annotated[list[str], operator.add]
    
    gaps: list[str]
    evaluation_rounds: int
    final_report: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 3: NODES (The Managers Tasks)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DecomposePlan(BaseModel):
    tasks: list[str] = Field(description="A dynamic list of highly specific, standalone research tasks. Create exactly as many tasks as needed based on the query complexity.")

def decompose_node(state: OrchestratorState):
    """
    Reads the user query and generates N sub-tasks. 
    If the human reviewed the last draft and generated feedback, it adjusts the plan.
    """
    llm = create_orchestrator_llm().with_structured_output(DecomposePlan)
    query = state["original_query"]
    feedback = state.get("human_feedback", "")
    
    prompt = f"User Query: {query}\n\nBreak this down into highly specific research tasks that isolated AI workers can research simultaneously. Create as many tasks as necessary to cover the complexity of the query."
    
    if feedback:
        print(f"\n💡 [Decompose] Received User Edit: '{feedback}'. Redrawing plan...")
        prompt += f"\n\n🚨 The user reviewed your previous plan and gave this feedback: {feedback}\nAdjust your tasks accordingly!"
    else:
        print(f"\n🧠 [Decompose] Drafting initial research tasks for query...")

    response = llm.invoke(prompt)
    print(f"   → Generated {len(response.tasks)} sub-tasks.")
    
    return {
        "research_plan": response.tasks,
        "human_feedback": "" # Wipe the feedback slate clean after applying it
    }

def plan_review_node(state: OrchestratorState):
    """
    DUMMY NODE. This node does nothing computationally. 
    It exists purely so we can hang an `interrupt_before` tag on it.
    This is where the LangGraph engine pauses and hands control to the terminal.
    """
    pass

class EvaluateGaps(BaseModel):
    is_complete: bool = Field(description="True if the findings fully answer the query. False if major info is missing.")
    gaps: list[str] = Field(description="List of specific questions that still need researching. Empty if is_complete is True.")

def evaluate_node(state: OrchestratorState):
    """
    The Reduce step. It reads all findings collected simultaneously by the workers.
    It checks if the user's original query was fully answered.
    """
    print(f"\n🔍 [Evaluate] All sub-agents returned. Reviewing {len(state.get('findings', []))} facts...")
    llm = create_orchestrator_llm().with_structured_output(EvaluateGaps)
    query = state["original_query"]
    findings = state.get("findings", [])
    
    findings_text = "\n".join([f"- {f.fact} (Confidence {f.confidence})" for f in findings])
    # Protect against empty findings across the board
    if not findings_text:
        findings_text = "No facts were found by the agents."

    prompt = f"User Query: {query}\n\nHere are the facts our agents found:\n{findings_text}\n\nAre there any critical gaps or missing information to fulfill the user query? If so, list them as clear research questions."
    
    response = llm.invoke(prompt)
    
    if not response.is_complete:
        print(f"   → Detected {len(response.gaps)} missing research gaps.")
    else:
        print(f"   → No gaps detected! Ready for synthesis.")

    return {
        "gaps": response.gaps if not response.is_complete else [],
        "evaluation_rounds": state.get("evaluation_rounds", 0) + 1
    }

def synthesize_node(state: OrchestratorState):
    """
    The final step. Collates all accurate facts into a polished markdown report.
    """
    print(f"\n✍️ [Synthesize] Writing final comprehensive report...")
    llm = create_orchestrator_llm() # Standard text output, no Pydantic needed formatting here
    query = state["original_query"]
    findings = state.get("findings", [])
    sources = list(set(state.get("sources", []))) # deduplicate
    
    findings_text = "\n".join([f"- {f.fact} (Source: {f.source_url})" for f in findings])
    
    prompt = f"User Query: {query}\n\nSynthesize a highly professional, comprehensive Markdown report using ONLY the following verified facts. DO NOT hallucinate. Group the report by logical headers.\n\nFacts:\n{findings_text}\n\nInclude a numbered reference list at the bottom matching the sources."
    
    response = llm.invoke(prompt)
    print(f"   → Report completed ({len(response.content)} characters).")

    return {"final_report": response.content}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 4: ROUTERS (Conditional Logic & Maps)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def route_human_approval(state: OrchestratorState):
    """
    After the pause finishes:
    If the user placed string feedback in the state, route back to decompose.
    Otherwise, map the plan into N parallel sub-agents using `.Send()`.
    """
    if state.get("human_feedback"):
        return "decompose_node"
    
    # User approved! Fire off exactly N sub-agents in parallel threads 🚀
    sends = []
    for task in state["research_plan"]:
        sub_initial_state = {
            "task": task,
            "messages": [HumanMessage(content=f"Research the following topic strictly and thoroughly:\n\n{task}")],
            "findings": [],
            "sources": [],
            "iterations": 0,
            "status": "running"
        }
        # Send(NODE_NAME, INITIAL_STATE)
        sends.append(Send("sub_agent", sub_initial_state))
        
    return sends

def route_evaluation(state: OrchestratorState):
    """
    Checks if there are gaps, and if we've tried patching them < 3 times.
    If yes, spawn new sub-agents JUST for the gaps. Otherwise proceed.
    """
    if len(state.get("gaps", [])) > 0 and state.get("evaluation_rounds", 0) < 3:
        print(f"🔄 [Router] Respawning sub-agents to research missing gaps!")
        sends = []
        for gap in state["gaps"]:
            sub_initial_state = {
                "task": f"Supplemental Gap Research: {gap}",
                "messages": [HumanMessage(content=f"Research strictly to fill this knowledge gap:\n\n{gap}")],
                "findings": [],
                "sources": [],
                "iterations": 0,
                "status": "running"
            }
            sends.append(Send("sub_agent", sub_initial_state))
        return sends
        
    return "synthesize_node"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 5: BUILD THE ORCHESTRATOR GRAPH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_lead_orchestrator():
    """
    Build the master graph that rules the sub-agents.
    """
    builder = StateGraph(OrchestratorState)
    
    # Add all standard python nodes
    builder.add_node("decompose_node", decompose_node)
    builder.add_node("plan_review_node", plan_review_node)
    
    # Sub-agent Node! We drop the entire graph from Step 2 directly in as a node module.
    builder.add_node("sub_agent", create_sub_agent_graph())
    
    builder.add_node("evaluate_node", evaluate_node)
    builder.add_node("synthesize_node", synthesize_node)
    
    # ── Define the flowchart edges ──
    builder.add_edge(START, "decompose_node")
    builder.add_edge("decompose_node", "plan_review_node")
    
    # From the review node, it conditionally maps to 1, or sends parallel maps (N)
    builder.add_conditional_edges("plan_review_node", route_human_approval)
    
    # When ANY sub-agent finishes, it flows safely into evaluate (LangGraph waits for all parallel sends automatically)
    builder.add_edge("sub_agent", "evaluate_node")
    
    # From evaluation, conditionally route to map more gaps, or end.
    builder.add_conditional_edges("evaluate_node", route_evaluation)
    
    # Once hitting synthesize, we're done.
    builder.add_edge("synthesize_node", END)
    
    # ── Memory ── 
    # To support Interrupts (pausing), LangGraph REQUIRES short term memory (checkpointer)
    # The state is written to memory, the console halts, and later it reads the memory again.
    memory = MemorySaver()
    
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["plan_review_node"] # 🛑 Pause execution exactly here
    )
