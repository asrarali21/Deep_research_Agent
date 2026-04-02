"""
Firecrawl Tool — Search + Scrape layer for the Deep Research Agent.

This is the TOOL LAYER (Step 1 of the build).
Sub-agents will call these two functions to gather information from the web.

Design principles:
  1. NEVER crash — errors return empty results, sub-agent keeps running.
  2. ONE tool does TWO jobs — search the web AND read a specific page.
  3. Clean return types — dicts with predictable keys, easy to consume.

Firecrawl SDK v4 API (verified against firecrawl-py 4.21.0):
  - app.search(query, limit=N)  → SearchData with .web list
  - app.scrape(url, formats=[...]) → Document with .markdown, .metadata
"""

import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp

# ── Load environment variables ──────────────────────────────────────────
load_dotenv()

# ── Initialize the Firecrawl client ONCE (reused across all calls) ─────
# Why a module-level singleton?
#   Every sub-agent will import this module. We want ONE client instance,
#   not a new one per function call. This saves memory and connection overhead.
_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUNCTION 1: search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def search(query: str, limit: int = 5) -> list[dict]:
    """
    Search the web using Firecrawl and return a list of results.

    Args:
        query: The search string (e.g., "LangGraph Send API parallel agents")
        limit: Max number of results to return (default 5, keeps cost low)

    Returns:
        A list of dicts, each with keys:
          - title   (str): Page title
          - url     (str): Page URL
          - snippet (str): Short description / excerpt

        On ANY error → returns an empty list [].
        The sub-agent sees "no results" and moves on. It never crashes.

    Example:
        >>> results = search("what is LangGraph")
        >>> results[0]
        {'title': 'LangGraph Docs', 'url': 'https://...', 'snippet': '...'}
    """
    try:
        # Call the Firecrawl Search API
        # Returns: SearchData (Pydantic model) with .web attribute
        #   .web = list of SearchResultWeb objects
        #   Each has: .title, .url, .description
        response = _app.search(query=query, limit=limit)

        results = []

        # response.web is a list of SearchResultWeb objects
        web_results = response.web or []

        for item in web_results:
            results.append({
                "title":   item.title or "No title",
                "url":     item.url or "",
                "snippet": item.description or "No snippet",
            })

        return results

    except Exception as e:
        # ⚠️ CRITICAL DESIGN DECISION:
        # We print the error for debugging but NEVER raise it.
        # A crashed tool = a crashed sub-agent = a wasted LLM call.
        # An empty result = the sub-agent tries a different query or moves on.
        print(f"[FirecrawlTool] Search failed: {e}")
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUNCTION 2: scrape
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scrape(url: str) -> dict:
    """
    Scrape a single URL and return its content as clean markdown.

    Args:
        url: The full URL to scrape (e.g., "https://docs.firecrawl.dev")

    Returns:
        A dict with keys:
          - url     (str):  The URL that was scraped
          - title   (str):  Page title (extracted by Firecrawl)
          - content (str):  Full page content as clean markdown
          - success (bool): True if we got content, False otherwise

        On ANY error → returns {url, title: "", content: "", success: False}.
        Same logic as search: fail gracefully, never crash.

    Example:
        >>> page = scrape("https://docs.firecrawl.dev")
        >>> page["success"]
        True
        >>> print(page["content"][:200])  # First 200 chars of markdown
    """
    try:
        # Request markdown format — that's what our LLM agents will consume.
        # Firecrawl handles JS rendering, cookie popups, paywalls, etc.
        # Returns: Document (Pydantic model) with .markdown, .metadata
        #   .metadata = DocumentMetadata with .title, .description, etc.
        response = _app.scrape(url, formats=["markdown"])

        markdown = response.markdown or ""
        title = ""

        # Extract title from metadata
        if response.metadata:
            title = response.metadata.title or ""

        return {
            "url":     url,
            "title":   title,
            "content": markdown,
            "success": bool(markdown),  # True only if we actually got content
        }

    except Exception as e:
        print(f"[FirecrawlTool] Scrape failed for {url}: {e}")
        return {
            "url":     url,
            "title":   "",
            "content": "",
            "success": False,
        }
