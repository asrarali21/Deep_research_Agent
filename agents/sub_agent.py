"""
Sub-Agent — A single autonomous researcher (LangGraph subgraph).

This is Step 2 of the build.

HOW IT WORKS (The ReAct Loop):
──────────────────────────────
  1. Orchestrator gives us a TASK (e.g., "Research quantum computing 2025")
  2. We enter a loop:
       THINK  → LLM decides what to do next (search? scrape? or done?)
       ACT    → We execute the tool the LLM chose
       OBSERVE → Tool results go back to LLM as a message
       ... repeat ...
  3. When LLM says "I'm done" → we package findings and return them

LangGraph models this as a StateGraph:
  Nodes:  reason (LLM thinks) → act (run tool) → finalize (package results)
  Edges:  reason → act (if LLM called a tool)
          reason → finalize (if LLM is done)
          act → reason (loop back to think again)

SAFETY LIMITS:
  - Max 8 iterations per sub-agent (prevents infinite loops)
  - Tools never crash (they return empty on error)
  - If anything unexpected happens, we finalize with whatever we have
"""

import os
import json
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ── LangGraph imports ───────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import ValidationError

# ── LangChain imports ───────────────────────────────────────────────────
from langchain_core.messages import (
    HumanMessage,      # Messages from the user (or orchestrator)
    AIMessage,          # Messages from the LLM
    SystemMessage,      # The system prompt (instructions for the LLM)
    ToolMessage,        # Results from tool execution
)
from langchain_openai import ChatOpenAI

# ── Our tools from Step 1 ──────────────────────────────────────────────
from tools.firecrawl_tool import search, scrape

# ── Load environment variables ──────────────────────────────────────────
load_dotenv()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1: STATE SCHEMA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# This defines WHAT DATA flows through the graph.
# Think of it as the sub-agent's "brain state" — everything it knows
# and tracks as it researches.
#
# The `Annotated[list, add_messages]` is special LangGraph syntax:
# it means "when updating messages, APPEND new ones to the list
# instead of replacing the whole list." This is how the LLM builds
# up its conversation history across iterations.

class SubAgentState(TypedDict):
    task: str                                        # What to research (from Orchestrator)
    messages: Annotated[list, add_messages]           # Conversation history (LLM memory)
    findings: list[dict]                              # Extracted facts with sources
    sources: list[str]                                # All URLs visited
    iterations: int                                   # Loop counter (safety)
    status: str                                       # "running" | "done" | "max_iterations"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2: CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_ITERATIONS = 8    # Hard limit: no sub-agent runs more than 8 loops

# ── The System Prompt ───────────────────────────────────────────────────
# This is the "personality" and "instructions" for the sub-agent LLM.
# It tells the LLM WHO it is and HOW to behave.

SYSTEM_PROMPT = """You are a focused research agent. Your job is to thoroughly research a specific topic using web search and page reading tools.

## Your Tools
1. **search(query)** — Search the web for information. Returns titles, URLs, and snippets.
2. **scrape(url)** — Read the full content of a specific webpage. Returns the page as markdown.
3. **SubmitFinalFindings** — Use this tool to submit your final factual findings when you are done.

## Your Research Process
1. Start by searching for the main topic
2. Read the most promising results using scrape
3. If you need more detail, search again with refined queries
4. Extract KEY FACTS with their source URLs
5. When you have gathered enough information (at least 3-5 solid facts from 2+ different sources), CALL the `SubmitFinalFindings` tool to finish.

## Rules
- Be THOROUGH but EFFICIENT — don't search for the same thing twice
- Each fact must have a real source URL from your research
- Confidence: 0.9+ = directly stated, 0.7-0.9 = strongly implied, 0.5-0.7 = inferred
- If search returns empty, try a different query — don't give up immediately
- Focus on FACTS, not opinions
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2.5: PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Finding(BaseModel):
    fact: str = Field(description="A clear, specific factual statement")
    source_url: str = Field(description="The exact source URL where this fact was found")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 (0.9+ = directly stated, 0.7-0.9 = strongly implied, 0.5-0.7 = inferred)")

class SubmitFinalFindings(BaseModel):
    """Use this tool to submit your final findings when you have gathered enough information from your search and scrape tools to fulfill the research task."""
    findings: list[Finding] = Field(description="List of factual findings")
    summary: str = Field(description="A 2-3 sentence summary of what you found")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 3: TOOL DEFINITIONS (for the LLM)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# The LLM doesn't call Python functions directly.
# Instead, we describe our tools in a format the LLM understands,
# and when it wants to use a tool, it outputs a "tool call" message.
# Then WE execute the actual function and feed the result back.
#
# This is the bridge between "LLM wants to search" and "search() runs."

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for information on a topic. Returns a list of results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape",
            "description": "Read the full content of a webpage. Returns the page content as clean markdown text. Use this after searching to read promising results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to scrape"
                    }
                },
                "required": ["url"]
            }
        }
    }
]

# ── Map of tool names to actual Python functions ───────────────────────
# When the LLM says "call search with query=X", we look up "search"
# in this dict and call the real function.

TOOL_MAP = {
    "search": search,
    "scrape": scrape,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 4: CREATE THE LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# We use Groq's Llama 3 model — fast and cheap, perfect for sub-agents.
# The .bind_tools() call tells the LLM "you have these tools available."
# After this, when the LLM wants to search, it will output a special
# tool_call message instead of plain text.

def create_llm():
    """Create and return the LLM with tools bound to it."""
    llm = ChatOpenAI(
        model="worker-model",     # Maps to liteLLM config.yaml
        api_key="litellm",        # Dummy key (not needed for localhost proxy)
        base_url="http://0.0.0.0:4000",
        temperature=0,            # 0 = deterministic (same input → same output)
    )
    # bind_tools = "Hey LLM, these tools exist, you can call them"
    llm_with_tools = llm.bind_tools(tools_for_llm + [SubmitFinalFindings])
    return llm_with_tools


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 5: GRAPH NODES (The 3 steps of the loop)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── NODE 1: reason ──────────────────────────────────────────────────────
# The "thinking" step. Sends all messages to the LLM.
# The LLM either:
#   a) Returns a tool_call → "I want to search/scrape something"
#   b) Returns text → "I'm done, here are my findings"

def reason_node(state: SubAgentState) -> dict:
    """
    Send the conversation history to the LLM and get its next action.

    This is the BRAIN of the sub-agent. Every iteration:
    1. We give the LLM ALL previous messages (its memory)
    2. The LLM decides: search more? read a page? or done?
    3. We return its response to be routed by the conditional edge
    """
    llm = create_llm()

    # Build the message list: system prompt + all previous messages
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # Call the LLM — this is where the AI "thinks"
    response = llm.invoke(messages)

    # Increment iteration counter
    new_iterations = state.get("iterations", 0) + 1

    print(f"\n🧠 [Reason] Iteration {new_iterations}")
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"   → Wants to call: {tc['name']}({tc['args']})")
    else:
        print(f"   → Done thinking. Returning findings.")

    # Return updates to the state
    # "messages" uses add_messages, so this APPENDS the response
    return {
        "messages": [response],
        "iterations": new_iterations,
    }


# ── NODE 2: act ─────────────────────────────────────────────────────────
# The "doing" step. Executes whatever tool the LLM requested.
# Takes the tool_call from the LLM's response, runs the actual
# Python function, and wraps the result as a ToolMessage.

def act_node(state: SubAgentState) -> dict:
    """
    Execute the tool(s) that the LLM requested.

    Flow:
    1. Get the last AI message (which contains tool_calls)
    2. For each tool_call, run the actual Python function
    3. Wrap results as ToolMessages (so the LLM can read them)
    4. Track any new URLs found in sources
    """
    # The last message is always the AI's response with tool_calls
    last_message = state["messages"][-1]
    tool_messages = []
    new_sources = list(state.get("sources", []))

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"   🔧 [Act] Executing: {tool_name}({tool_args})")

        # Look up the actual function and call it
        tool_fn = TOOL_MAP.get(tool_name)

        if tool_fn:
            try:
                # SPECIAL TOOL: SubmitFinalFindings requires Pydantic Enforced Validation before we allow it to pass!
                if tool_name in ("SubmitFinalFindings", "submit_final_findings"):
                    # Validate the raw LLM JSON arguments instantly against the Pydantic logic!
                    validated_data = SubmitFinalFindings(**tool_args)
                    result = "Findings securely stored via Pydantic."
                else:
                    # Normal search or scrape tool
                    result = tool_fn(**tool_args)

                # Track URLs we've visited
                if tool_name == "scrape" and isinstance(result, dict):
                    new_sources.append(result.get("url", ""))
                elif tool_name == "search" and isinstance(result, list):
                    for r in result:
                        new_sources.append(r.get("url", ""))

                # Convert result to string for the LLM
                result_str = json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
                print(f"   ✅ [Act] Got result ({len(result_str)} chars)")

            except ValidationError as e:
                # 🚨 PYDANTIC FIRED AN ERROR! The LLM hallucinated the JSON schema!
                # We return the exact Pydantic Error straight back to the LLM so it learns and fixes its mistake on the next iteration!
                result_str = f"Pydantic Validation Error! Your arguments did not match the schema: {str(e)}\nPlease rewrite your tool call using the correct structure."
                print(f"   ❌ [Act] {result_str}")
            except Exception as e:
                result_str = f"Execution Error: {str(e)}"
                print(f"   ❌ [Act] {result_str}")
        else:
            # Fake tool for the final Pydantic check (if TOOL_MAP missed it)
            if tool_name in ("SubmitFinalFindings", "submit_final_findings"):
                try:
                    SubmitFinalFindings(**tool_args)
                    result_str = "Findings successfully recorded."
                    print(f"   ✅ [Act] Pydantic Validated.")
                except ValidationError as e:
                    result_str = f"Pydantic Validation Error! {str(e)}"
                    print(f"   ❌ [Act] {result_str}")
            else:
                result_str = f"Error: Unknown tool '{tool_name}'"
                print(f"   ❌ [Act] {result_str}")

        # Wrap as ToolMessage — the LLM needs this format to understand
        # that this is a response to ITS tool call (matched by tool_call_id)
        tool_messages.append(
            ToolMessage(
                content=result_str,
                tool_call_id=tool_call["id"],  # Links response to the request
            )
        )

    return {
        "messages": tool_messages,
        "sources": new_sources,
    }


# ── NODE 3: finalize ───────────────────────────────────────────────────
# The "packaging" step. Takes the LLM's final text response
# and extracts structured findings from it.

def finalize_node(state: SubAgentState) -> dict:
    """
    Extract structured findings from the LLM's final response using Pydantic structure.
    """
    last_message = state["messages"][-1]
    
    findings = []
    summary = "No summary provided."
    status = "done"

    if state.get("iterations", 0) >= MAX_ITERATIONS:
        status = "max_iterations"
        print(f"   ⚠️  Hit max iterations ({MAX_ITERATIONS})")
    else:
        print(f"   ✅ Completed in {state.get('iterations', 0)} iterations")

    # If the LLM successfully used the Pydantic tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tc in last_message.tool_calls:
            if tc["name"] in ("SubmitFinalFindings", "submit_final_findings"):
                try:
                    # 1. STRICT Pydantic Enforcement! No manual type overwriting!
                    validated_output = SubmitFinalFindings(**tc["args"])
                    
                    # 2. Extract perfectly structured fields based purely on Pydantic
                    # Pydantic guarantees these are the exact correct types (List[Finding], str)
                    findings = validated_output.findings
                    summary = validated_output.summary
                    
                    print(f"\n📋 [Finalize] Extracted {len(findings)} findings via Strict Pydantic Validation\n")
                    
                    # Create a fake ToolMessage to satisfy LangGraph's requirement that every tool call has a response
                    tool_msg = ToolMessage(content="Findings submitted gracefully.", tool_call_id=tc["id"])
                    
                    return {
                        "messages": [tool_msg],
                        "findings": findings,
                        "status": status,
                    }
                except ValidationError as e:
                    # If this happens, it means the graph exhausted max_iterations while repeatedly trying to fix Pydantic errors.
                    print(f"\n📋 [Finalize] Pydantic Validation critically failed after max loops: {e}")
                    pass

    # Fallback: if it just returned text instead of calling the tool
    content = last_message.content if hasattr(last_message, "content") else str(last_message)
    print(f"\n📋 [Finalize] Fallback: LLM didn't use Pydantic tool, returning raw text")
    findings = [Finding(
        fact=content[:500],
        source_url="parsed_from_conversation",
        confidence=0.5
    )]

    return {
        "findings": findings,
        "status": status,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 6: ROUTER (Conditional Edge)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# After the "reason" node runs, we need to decide:
# → Did the LLM call a tool? → Go to "act"
# → Did the LLM give a final answer? → Go to "finalize"
# → Did we hit max iterations? → Go to "finalize"
#
# This function returns the NAME of the next node to execute.

def should_continue(state: SubAgentState) -> str:
    """
    Route after the 'reason' node.
    Returns the name of the next node: "act" or "finalize"
    """
    # Safety: if we've hit max iterations, stop no matter what
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        print(f"   ⚠️  Max iterations reached, forcing finalize")
        return "finalize"

    # Check if the LLM wants to call a tool
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Check if the LLM is calling the final findings tool
        for tc in last_message.tool_calls:
            if tc["name"] in ("SubmitFinalFindings", "submit_final_findings"):
                try:
                    # Only proceed to finalize if Pydantic actually validates!
                    SubmitFinalFindings(**tc["args"])
                    return "finalize"
                except ValidationError:
                    # If validation fails, route it to ACT! 
                    # ACT will execute the Validation Error and feed it back to REASON.
                    return "act"
        return "act"       # LLM wants to use search/scrape → go execute it
    else:
        return "finalize"  # LLM gave text response → it's done


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 7: BUILD THE GRAPH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# This is where we wire everything together into a LangGraph StateGraph.
# It's like building a flowchart:
#
#   START → reason → (tool_calls?) → act → reason → ... → finalize → END
#                     (no tool_calls?) ──────────────────→ finalize → END

def create_sub_agent_graph():
    """
    Build and compile the sub-agent graph.

    Returns a compiled graph that can be invoked with:
        result = graph.invoke({"task": "...", "messages": [...], ...})
    """
    # Create the graph with our state schema
    graph = StateGraph(SubAgentState)

    # Add the 3 nodes
    graph.add_node("reason", reason_node)
    graph.add_node("act", act_node)
    graph.add_node("finalize", finalize_node)

    # Wire the edges:

    # 1. START → reason (always start by thinking)
    graph.add_edge(START, "reason")

    # 2. reason → act OR finalize (conditional — depends on LLM output)
    graph.add_conditional_edges("reason", should_continue)

    # 3. act → reason (after executing a tool, think again)
    graph.add_edge("act", "reason")

    # 4. finalize → END (we're done)
    graph.add_edge("finalize", END)

    # Compile = "lock in the graph, make it runnable"
    return graph.compile()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 8: CONVENIENCE FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# A simple wrapper so the Orchestrator (Step 3) can just call:
#   result = run_sub_agent("research quantum computing")

def run_sub_agent(task: str) -> dict:
    """
    Run a sub-agent on a specific research task.

    Args:
        task: The research task (e.g., "Research quantum computing breakthroughs 2025")

    Returns:
        dict with keys:
          - findings: list of {fact, source_url, confidence}
          - sources: list of all URLs visited
          - iterations: how many reasoning loops
          - status: "done" | "max_iterations"
    """
    graph = create_sub_agent_graph()

    # Set up the initial state
    initial_state = {
        "task": task,
        "messages": [
            HumanMessage(content=f"Research the following topic thoroughly:\n\n{task}")
        ],
        "findings": [],
        "sources": [],
        "iterations": 0,
        "status": "running",
    }

    print(f"\n{'='*60}")
    print(f"🚀 Sub-Agent Starting")
    print(f"   Task: {task}")
    print(f"{'='*60}")

    # Run the graph — this executes the full ReAct loop
    final_state = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"🏁 Sub-Agent Finished")
    print(f"   Status: {final_state.get('status', 'unknown')}")
    print(f"   Iterations: {final_state.get('iterations', 0)}")
    print(f"   Findings: {len(final_state.get('findings', []))}")
    print(f"   Sources: {len(final_state.get('sources', []))}")
    print(f"{'='*60}")

    return {
        "findings": final_state.get("findings", []),
        "sources": final_state.get("sources", []),
        "iterations": final_state.get("iterations", 0),
        "status": final_state.get("status", "unknown"),
    }
