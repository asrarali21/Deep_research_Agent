import unittest

from services.source_quality import is_generic_low_signal_result, score_search_result


class SourceQualityTests(unittest.TestCase):
    def test_generic_writing_results_flagged_for_normal_research_queries(self):
        self.assertTrue(
            is_generic_low_signal_result(
                "AI agents market size and pricing in 2026",
                "How to Write Introductions for Research Papers",
                "A complete writing guide for academic introductions.",
                "https://example.com/how-to-write-introductions",
            )
        )

    def test_writing_queries_do_not_flag_writing_guides(self):
        self.assertFalse(
            is_generic_low_signal_result(
                "How to write a research paper introduction",
                "How to Write Introductions for Research Papers",
                "A complete writing guide for academic introductions.",
                "https://example.com/how-to-write-introductions",
            )
        )

    def test_generic_writing_results_are_scored_below_relevant_domain_results(self):
        low_signal_score = score_search_result(
            "AI agents market size and pricing in 2026",
            "How to Write Introductions for Research Papers",
            "A complete writing guide for academic introductions.",
            "https://example.com/how-to-write-introductions",
        )
        relevant_score = score_search_result(
            "AI agents market size and pricing in 2026",
            "AI agents market pricing and enterprise adoption in 2026",
            "Pricing benchmarks, adoption rates, and contract sizes for enterprise deployments.",
            "https://example.com/ai-agents-pricing-2026",
        )

        self.assertLess(low_signal_score, relevant_score)
