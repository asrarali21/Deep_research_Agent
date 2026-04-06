import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools import firecrawl_tool


class FirecrawlToolCacheTests(unittest.TestCase):
    def setUp(self):
        firecrawl_tool._cache.clear()

    def test_search_uses_normalized_cache_key(self):
        response = SimpleNamespace(
            web=[SimpleNamespace(title="Example", url="https://example.com", description="Snippet")]
        )
        with patch.object(firecrawl_tool._app, "search", return_value=response) as mock_search:
            first = firecrawl_tool.search("Example Query", limit=3)
            second = firecrawl_tool.search("  example   query  ", limit=3)

        self.assertEqual(first, second)
        self.assertEqual(mock_search.call_count, 1)

    def test_scrape_uses_normalized_url_cache_key(self):
        response = SimpleNamespace(markdown="Body", metadata=SimpleNamespace(title="Example"))
        with patch.object(firecrawl_tool._app, "scrape", return_value=response) as mock_scrape:
            first = firecrawl_tool.scrape("https://example.com/path/")
            second = firecrawl_tool.scrape("https://example.com/path")

        self.assertEqual(first, second)
        self.assertEqual(mock_scrape.call_count, 1)
