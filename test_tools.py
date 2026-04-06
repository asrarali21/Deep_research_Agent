"""
Test script for the Firecrawl Tool.

Run this from the project root:
    python test_tools.py

This tests BOTH functions:
  1. search() — Does a real web search, prints results
  2. scrape() — Reads a real webpage, prints a preview

It also tests error handling:
  3. scrape() with a bad URL — should return success=False, not crash
"""

from tools.firecrawl_tool import search, scrape

# ── Separator for readable output ──────────────────────────────────────
def divider(label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    divider("TEST 1: search('latest advancements in AI agents 2025')")

    results = search("latest advancements in AI agents 2025", limit=3)

    if results:
        print(f"✅ Got {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['title']}")
            print(f"     URL:     {r['url']}")
            print(f"     Snippet: {r['snippet'][:120]}...")
            print()
    else:
        print("❌ No results returned (check your API key or network)")

    divider("TEST 2: scrape('https://docs.firecrawl.dev')")

    page = scrape("https://docs.firecrawl.dev")

    print(f"  URL:     {page['url']}")
    print(f"  Title:   {page['title']}")
    print(f"  Success: {page['success']}")
    print(f"  Content preview ({len(page['content'])} chars total):")
    print(f"  {'-'*50}")
    print(f"  {page['content'][:500]}")
    print(f"  {'-'*50}")

    if page["success"]:
        print("\n✅ Scrape succeeded!")
    else:
        print("\n❌ Scrape failed (check URL or API key)")

    divider("TEST 3: scrape('https://thisurldoesnotexist12345.com') — error handling")

    bad_page = scrape("https://thisurldoesnotexist12345.com")

    print(f"  Success: {bad_page['success']}")
    print(f"  Content: '{bad_page['content']}'")

    if not bad_page["success"] and bad_page["content"] == "":
        print("\n✅ Error handled gracefully! No crash, empty result returned.")
    else:
        print("\n❌ Unexpected: should have returned success=False with empty content")

    divider("SUMMARY")
    print("  If all 3 tests show ✅, your Firecrawl tool layer is READY.")
    print("  Next step: Build the Sub-Agent (Step 2).")
    print()
