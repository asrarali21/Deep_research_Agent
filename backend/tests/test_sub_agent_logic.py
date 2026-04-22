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

    def test_assess_submission_quality_rejects_unscraped_and_weak_sources(self):
        state = {
            "trace_id": "trace-1",
            "task": "task",
            "messages": [],
            "working_summary": "",
            "findings": [],
            "evidence_cards": [],
            "sources": ["https://example.com/source-a"],
            "discovered_sources": ["https://example.com/source-a", "https://duckduckgo.com"],
            "seen_source_urls": ["https://example.com/source-a"],
            "coverage_tags": [],
            "completed_tasks": [],
            "iterations": 1,
            "status": "running",
        }
        submitted = sub_agent.SubmitFinalFindings(
            findings=[
                sub_agent.Finding(
                    fact="Fact",
                    source_url="https://duckduckgo.com",
                    confidence=0.9,
                )
            ],
            evidence_cards=[
                sub_agent.EvidenceCard(
                    claim="Claim",
                    source_url="https://duckduckgo.com",
                    source_title="DuckDuckGo",
                    excerpt="Snippet",
                    section_tag="general",
                    source_type="commercial",
                    authority_score=1,
                    confidence=0.9,
                )
            ],
            coverage_tags=["general"],
            summary="Summary",
        )

        issues, valid_findings, valid_cards, coverage_tags = sub_agent.assess_submission_quality(state, submitted)

        self.assertTrue(issues)
        self.assertEqual(valid_findings, [])
        self.assertEqual(valid_cards, [])
        self.assertEqual(coverage_tags, ["general"])

    def test_finalize_node_does_not_fabricate_evidence_when_submission_is_missing(self):
        state = {
            "trace_id": "trace-1",
            "task": "task",
            "messages": [AIMessage(content="Here is a narrative summary without tool output.")],
            "working_summary": "Search results and scraps of notes",
            "findings": [],
            "evidence_cards": [],
            "sources": ["https://example.com/source-a"],
            "discovered_sources": ["https://example.com/source-a"],
            "seen_source_urls": ["https://example.com/source-a"],
            "coverage_tags": [],
            "completed_tasks": [],
            "iterations": 1,
            "status": "running",
        }

        result = sub_agent.finalize_node(state)

        self.assertEqual(result["findings"], [])
        self.assertEqual(result["evidence_cards"], [])
        self.assertEqual(result["coverage_tags"], [])
        self.assertEqual(result["sources"], ["https://example.com/source-a"])
        self.assertEqual(result["completed_tasks"], ["task"])
        self.assertEqual(result["status"], "insufficient_evidence")

    def test_assess_submission_quality_accepts_canonical_url_variants_as_verified(self):
        state = {
            "trace_id": "trace-1",
            "task": "simple task about example",
            "messages": [],
            "working_summary": "",
            "findings": [],
            "evidence_cards": [],
            "sources": ["https://example.com/source-a"],
            "discovered_sources": ["https://example.com/source-a"],
            "seen_source_urls": ["https://example.com/source-a"],
            "coverage_tags": [],
            "completed_tasks": [],
            "iterations": 1,
            "status": "running",
        }
        submitted = sub_agent.SubmitFinalFindings(
            findings=[
                sub_agent.Finding(
                    fact="Fact",
                    source_url="https://example.com/source-a",
                    confidence=0.9,
                )
            ],
            evidence_cards=[
                sub_agent.EvidenceCard(
                    claim="Claim",
                    source_url="https://example.com/source-a?utm_source=newsletter&utm_medium=email",
                    source_title="Example",
                    excerpt="Excerpt text",
                    section_tag="general",
                    source_type="guideline",
                    authority_score=9,
                    confidence=0.9,
                )
            ],
            coverage_tags=["general"],
            summary="Summary",
        )

        issues, valid_findings, valid_cards, _ = sub_agent.assess_submission_quality(state, submitted)

        # Should not lose the evidence card due to URL mismatch and should rewrite to scraped URL.
        self.assertEqual(valid_findings[0]["source_url"], "https://example.com/source-a")
        self.assertEqual(len(valid_cards), 1)
        self.assertEqual(valid_cards[0]["source_url"], "https://example.com/source-a")
        self.assertEqual(valid_cards[0].get("verification_status"), "verified")
        self.assertIsInstance(issues, list)
