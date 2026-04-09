import unittest

from langchain_core.messages import AIMessage

from agents import sub_agent


class SubAgentLogicTests(unittest.TestCase):
    def test_search_results_do_not_mark_urls_as_scraped_sources(self):
        original_tool = sub_agent.TOOL_MAP["SearchTool"]
        try:
            sub_agent.TOOL_MAP["SearchTool"] = lambda **_: [
                {
                    "title": "Example Result",
                    "url": "https://example.com/article",
                    "snippet": "Example snippet",
                }
            ]
            state = {
                "trace_id": "trace-1",
                "task": "task",
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "SearchTool", "args": {"query": "query"}, "id": "tool-1"}],
                    )
                ],
                "working_summary": "",
                "findings": [],
                "evidence_cards": [],
                "sources": [],
                "discovered_sources": [],
                "seen_source_urls": [],
                "coverage_tags": [],
                "completed_tasks": [],
                "iterations": 1,
                "status": "running",
            }

            result = sub_agent.act_node(state)

            self.assertEqual(result["seen_source_urls"], [])
            self.assertEqual(result["sources"], [])
            self.assertEqual(result["discovered_sources"], ["https://example.com/article"])
            self.assertIn("https://example.com/article", result["working_summary"])
        finally:
            sub_agent.TOOL_MAP["SearchTool"] = original_tool

    def test_normalize_section_tag_maps_common_medical_aliases(self):
        self.assertEqual(sub_agent.normalize_section_tag("Cardiac Rehabilitation"), "cardiac_rehab")
        self.assertEqual(sub_agent.normalize_section_tag("salt intake"), "sodium")
