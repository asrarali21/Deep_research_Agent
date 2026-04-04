"""
Test script for the Lead Orchestrator (Step 3).

Run this from the project root:
    python test_orchestrator.py

This demonstrates:
  1. The Orchestrator breaking the query into N tasks.
  2. The graph pausing (interrupt).
  3. The terminal asking for your Y / N (+ edits) input.
  4. LangGraph launching all N Sub-Agents in parallel using Map-Reduce.
  5. The evaluation and final Markdown synthesis.
"""

from agents.lead_orchestrator_agent import create_lead_orchestrator

def divider(title=""):
    print("\n" + "="*80)
    if title:
        print(f" {title.upper()} ".center(80, "="))
    print("="*80 + "\n")

# 1. Initialize the graph
graph = create_lead_orchestrator()

# Checkpointing requires a thread ID to track the specific conversation memory
config = {"configurable": {"thread_id": "test_session_1"}}

# Standard query
# query = "What is the global market size for AI agents in 2025, what are the top 3 deployment challenges, and how does multi-agent architecture solve the latency problems of single agents?"
query = "What is the global market size for AI agents in 2025, and what are the top deployment challenges?"

divider("STARTING DEEP RESEARCH ORCHESTRATOR")
print(f"User Query: {query}")

# 2. Run the graph up to the breakpoint
initial_state = {
    "original_query": query,
    "research_plan": [],
    "human_feedback": "",
    "findings": [],
    "sources": [],
    "gaps": [],
    "evaluation_rounds": 0,
    "final_report": ""
}

# invoke() runs until completion, OR until hit an interrupt_before node
print("\n[Engine] Starting graph execution...")
for chunk in graph.stream(initial_state, config):
    # LangGraph streams updates. The chunk is a dict of {node_name: {state_update}}
    for key, value in chunk.items():
        if key == "decompose_node":
            plan = value.get("research_plan", [])
            print(f"\n[Engine] Decompose Node finished. Generated Plan ({len(plan)} tasks):")
            for i, task in enumerate(plan, 1):
                print(f"  {i}. {task}")

# 3. Handle the Human-In-The-Loop Interrupt
# We check if the graph is waiting (suspended)
state_snapshot = graph.get_state(config)

if state_snapshot.next:
    print(f"\n[Engine] Graph hit breakpoint. Next node in queue is: {state_snapshot.next}")
    divider("HUMAN APPROVAL REQUIRED")
    
    print("Review the Draft Plan above.")
    print("If you approve, type 'y' or press Enter.")
    print("If you want changes, type your feedback (e.g., 'Also research X').")
    
    user_input = input("\nYour response: ").strip()
    
    # Update the graph state with the human feedback
    if user_input and user_input.lower() not in ["y", "yes", "approve"]:
        print("\n[Engine] Updating state with human feedback and resuming graph...")
        graph.update_state(config, {"human_feedback": user_input})
    else:
        print("\n[Engine] Human approved the plan! Proceeding to execute N parallel agents...")
        # To proceed without changing anything, we just tell the graph to continue with None
        pass

    # Resume the graph from where it paused!
    for chunk in graph.stream(None, config):
        pass # The internal print statements inside the nodes will show progress

divider("FINAL REPORT")
# When finished, get the final state
final_state = graph.get_state(config).values

report = final_state.get("final_report", "No report generated.")
print(report)

print("\n" + "="*80)
print(f"Total Sources Pinged: {len(list(set(final_state.get('sources', []))))}")
print(f"Total Facts Combined: {len(final_state.get('findings', []))}")
print(f"Evaluation Rounds (Gap searches): {final_state.get('evaluation_rounds', 0)}")
print("="*80 + "\n")
