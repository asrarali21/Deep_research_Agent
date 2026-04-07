import os
import time
from urllib.parse import urlsplit, urlunsplit

import trafilatura
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain_community.tools import DuckDuckGoSearchRun

from services.config import get_settings

load_dotenv()

_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
_ddg_search = DuckDuckGoSearchRun()
_cache: dict[str, tuple[float, object]] = {}


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _normalize_url(url: str) -> str:
    split = urlsplit(url.strip())
    normalized_path = split.path.rstrip("/") or "/"
    return urlunsplit((split.scheme.lower(), split.netloc.lower(), normalized_path, split.query, ""))


def _cache_get(key: str):
    if key not in _cache:
        return None
    expires_at, value = _cache[key]
    if expires_at <= time.time():
        _cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value, ttl_seconds: int) -> None:
    _cache[key] = (time.time() + ttl_seconds, value)


def search(query: str, limit: int | None = None) -> list[dict]:
    settings = get_settings()
    limit = limit or settings.search_result_limit
    normalized_query = _normalize_query(query)
    cache_key = f"search:{normalized_query}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    results: list[dict] = []
    try:
        response = _app.search(query=query, limit=limit)
        for item in response.web or []:
            results.append(
                {
                    "title": item.title or "No title",
                    "url": item.url or "",
                    "snippet": item.description or "No snippet",
                }
            )
    except Exception:
        results = []

    if not results:
        try:
            raw_res = _ddg_search.run(query)
            results = [
                {
                    "title": "DuckDuckGo Result",
                    "url": "https://duckduckgo.com",
                    "snippet": raw_res[:1000],
                }
            ]
        except Exception:
            results = []

    deduped_results: list[dict] = []
    seen_urls = set()
    for item in results:
        url = item.get("url", "")
        key = url or item.get("title", "").lower()
        if key in seen_urls:
            continue
        seen_urls.add(key)
        deduped_results.append(item)
        if len(deduped_results) >= limit:
            break

    _cache_set(cache_key, deduped_results, settings.search_cache_ttl_seconds)
    return deduped_results


def scrape(url: str) -> dict:
    settings = get_settings()
    normalized_url = _normalize_url(url)
    cache_key = f"scrape:{normalized_url}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    payload = {
        "url": url,
        "title": "",
        "content": "",
        "success": False,
    }

    try:
        response = _app.scrape(url, formats=["markdown"])
        if response and response.markdown:
            payload = {
                "url": url,
                "title": response.metadata.title if response.metadata else "",
                "content": response.markdown,
                "success": True,
            }
    except Exception:
        payload = payload

    if not payload["success"]:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                content = trafilatura.extract(downloaded, include_links=True, include_images=False)
                if content:
                    payload = {
                        "url": url,
                        "title": "Extracted Content",
                        "content": content,
                        "success": True,
                    }
        except Exception:
            payload = payload

    _cache_set(cache_key, payload, settings.scrape_cache_ttl_seconds)
    return payload
