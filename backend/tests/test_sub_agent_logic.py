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
                "sources": [],
                "seen_source_urls": [],
                "iterations": 1,
                "status": "running",
            }

            result = sub_agent.act_node(state)

            self.assertEqual(result["seen_source_urls"], [])
            self.assertEqual(result["sources"], [])
            self.assertIn("https://example.com/article", result["working_summary"])
        finally:
            sub_agent.TOOL_MAP["SearchTool"] = original_tool
