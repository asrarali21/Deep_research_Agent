import unittest

from services.report_quality import GOLDEN_RESEARCH_PROMPTS, score_report_quality


class ReportQualityTests(unittest.TestCase):
    def test_golden_prompt_set_covers_core_report_types(self):
        joined = " ".join(GOLDEN_RESEARCH_PROMPTS).lower()

        self.assertIn("market", joined)
        self.assertIn("compare", joined)
        self.assertIn("clinical", joined)
        self.assertIn("compliance", joined)

    def test_score_report_quality_rewards_deep_report_signals(self):
        report = """
# Report

## Executive Summary
Summary with [E1].

## Key Findings
| Area | Finding |
|---|---|
| Market | $7.2 billion in 2025 [E1] |

## Market Size
The market reached $7.2 billion in 2025 and could expand by 31% by 2026 [E1].

## Pricing
Enterprise plans exceed $100,000 annually [E2].

## Timeline
The chronology from 2024 to 2026 shows adoption accelerating [E3].

## References
1. [Source A](https://example.com/a) (R1; news; evidence=E1)
2. [Source B](https://example.com/b) (R2; news; evidence=E2)
"""
        score = score_report_quality(
            report,
            [
                {"url": "https://example.com/a"},
                {"url": "https://example.com/b"},
            ],
        )

        self.assertGreaterEqual(score.score, 50)
        self.assertTrue(score.has_table)
        self.assertTrue(score.has_timeline_or_chronology)
        self.assertEqual(score.unsupported_reference_count, 0)


if __name__ == "__main__":
    unittest.main()
