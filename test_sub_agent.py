"""
Test script for the Sub-Agent (Step 2).

Run this from the project root:
    python test_sub_agent.py

This runs a SINGLE sub-agent with a real research task.
You'll see the full ReAct loop in action:
  🧠 [Reason] — LLM thinking
  🔧 [Act]    — Tool executing
  📋 [Finalize] — Packaging results

Watch the iteration numbers — that's the loop in real time!
"""

from agents.sub_agent import run_sub_agent
import json


# ── Run the sub-agent with a real research task ────────────────────────
print("\n" + "🔬" * 30)
print("  SUB-AGENT TEST — Live Research")
print("🔬" * 30)

result = run_sub_agent("What are the latest breakthroughs in AI agents and multi-agent systems in 2025?")


# ── Display the results ────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("  FINAL RESULTS")
print("=" * 60)

print(f"\n📊 Status: {result['status']}")
print(f"🔄 Iterations: {result['iterations']}")
print(f"🔗 Sources visited: {len(result['sources'])}")
print(f"📝 Findings: {len(result['findings'])}")

print("\n" + "-" * 60)
print("  FINDINGS (what the sub-agent discovered):")
print("-" * 60)

for i, finding in enumerate(result["findings"], 1):
    if isinstance(finding, dict):
        print(f"\n  {i}. {finding.get('fact', 'N/A')}")
        print(f"     Source: {finding.get('source_url', 'N/A')}")
        print(f"     Confidence: {finding.get('confidence', 'N/A')}")
    else:
        print(f"\n  {i}. {finding}")

print("\n" + "-" * 60)
print("  ALL SOURCES:")
print("-" * 60)
# Deduplicate and show unique sources
unique_sources = list(set(s for s in result["sources"] if s))
for i, src in enumerate(unique_sources, 1):
    print(f"  {i}. {src}")

print("\n" + "=" * 60)
print("  ✅ Sub-Agent test complete!")
print("  If you see findings above, Step 2 is WORKING.")
print("  Next: Step 3 — Build the Lead Orchestrator.")
print("=" * 60 + "\n")
