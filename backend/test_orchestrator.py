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

import asyncio

from agents.lead_orchestrator_agent import create_lead_orchestrator
from services.runtime import initialize_runtime

def divider(title=""):
    print("\n" + "="*80)
    if title:
        print(f" {title.upper()} ".center(80, "="))
    print("="*80 + "\n")

async def main():
    await initialize_runtime()
    graph = create_lead_orchestrator()
    config = {"configurable": {"thread_id": "test_session_1"}}
    query = "What is the global market size for AI agents in 2025, and what are the top deployment challenges?"

    divider("STARTING DEEP RESEARCH ORCHESTRATOR")
    print(f"User Query: {query}")

    initial_state = {
        "thread_id": "test_session_1",
        "original_query": query,
        "depth_profile": "deep",
        "research_contract": {},
        "depth_budget": {},
        "research_plan": [],
        "required_sections": [],
        "pending_tasks": [],
        "current_batch": [],
        "human_feedback": "",
        "findings": [],
        "evidence_cards": [],
        "sources": [],
        "discovered_sources": [],
        "coverage_tags": [],
        "completed_tasks": [],
        "gaps": [],
        "quality_summary": "",
        "evaluation_rounds": 0,
        "targeted_gap_rounds": 0,
        "outline_sections": [],
        "section_packets": [],
        "priority_sections": [],
        "section_drafts": {},
        "section_verifications": {},
        "report_reference_urls": [],
        "structured_references": [],
        "final_report": "",
    }

    print("\n[Engine] Starting graph execution...")
    async for chunk in graph.astream(initial_state, config):
        for key, value in chunk.items():
            if key == "decompose_node":
                plan = value.get("research_plan", [])
                print(f"\n[Engine] Decompose Node finished. Generated Plan ({len(plan)} tasks):")
                for i, task in enumerate(plan, 1):
                    print(f"  {i}. {task}")

    state_snapshot = await graph.aget_state(config)

    if state_snapshot.next:
        print(f"\n[Engine] Graph hit breakpoint. Next node in queue is: {state_snapshot.next}")
        divider("HUMAN APPROVAL REQUIRED")
        print("Review the Draft Plan above.")
        print("If you approve, type 'y' or press Enter.")
        print("If you want changes, type your feedback (e.g., 'Also research X').")

        user_input = input("\nYour response: ").strip()
        if user_input and user_input.lower() not in ["y", "yes", "approve"]:
            print("\n[Engine] Updating state with human feedback and resuming graph...")
            await graph.aupdate_state(config, {"human_feedback": user_input})
        else:
            print("\n[Engine] Human approved the plan! Proceeding to execute batched agents...")

        async for _chunk in graph.astream(None, config):
            pass

    divider("FINAL REPORT")
    final_state = (await graph.aget_state(config)).values
    report = final_state.get("final_report", "No report generated.")
    print(report)
    print("\n" + "=" * 80)
    print(f"Total Sources Pinged: {len(list(set(final_state.get('sources', []))))}")
    print(f"Total Facts Combined: {len(final_state.get('findings', []))}")
    print(f"Evaluation Rounds (Gap searches): {final_state.get('evaluation_rounds', 0)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
