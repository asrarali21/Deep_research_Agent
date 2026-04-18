import os
import time

from dotenv import load_dotenv

try:
    import trafilatura
except ImportError:  # pragma: no cover - optional dependency in local/dev environments
    trafilatura = None

try:
    from firecrawl import FirecrawlApp
except ImportError:  # pragma: no cover - optional dependency in local/dev environments
    FirecrawlApp = None

try:
    from langchain_community.tools import DuckDuckGoSearchResults
except ImportError:  # pragma: no cover - optional dependency in local/dev environments
    DuckDuckGoSearchResults = None

from services.config import get_settings
from services.source_quality import (
    compute_authority_score,
    infer_source_type,
    is_generic_low_signal_result,
    is_reference_usable,
    looks_like_homepage,
    normalize_url,
    score_search_result,
)

load_dotenv()

_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY")) if FirecrawlApp and os.getenv("FIRECRAWL_API_KEY") else None
_ddg_search = DuckDuckGoSearchResults(output_format="list", num_results=8) if DuckDuckGoSearchResults else None
_cache: dict[str, tuple[float, object]] = {}


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


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


def _coerce_search_result(title: str, url: str, snippet: str, query: str) -> dict | None:
    if not url or not is_reference_usable(url):
        return None

    normalized = normalize_url(url)
    if is_generic_low_signal_result(query, title, snippet, normalized):
        return None

    source_type = infer_source_type(normalized)
    authority_score = compute_authority_score(normalized, source_type)
    relevance_score = score_search_result(query, title, snippet, normalized)
    return {
        "title": title or "No title",
        "url": normalized,
        "snippet": snippet or "No snippet",
        "source_type": source_type,
        "authority_score": authority_score,
        "relevance_score": relevance_score,
        "is_homepage": looks_like_homepage(normalized),
    }


def _collect_firecrawl_results(query: str, limit: int) -> list[dict]:
    results: list[dict] = []
    if _app is None:
        return results
    try:
        response = _app.search(query=query, limit=max(limit * 2, limit))
        for item in response.web or []:
            coerced = _coerce_search_result(
                item.title or "No title",
                item.url or "",
                item.description or "No snippet",
                query,
            )
            if coerced is not None:
                results.append(coerced)
    except Exception:
        return []
    return results


def _collect_ddg_results(query: str) -> list[dict]:
    results: list[dict] = []
    if _ddg_search is None:
        return results
    try:
        raw_results = _run_ddg_search(query)
        for item in raw_results or []:
            coerced = _coerce_search_result(
                item.get("title", "No title"),
                item.get("link", "") or item.get("url", ""),
                item.get("snippet", "") or item.get("body", ""),
                query,
            )
            if coerced is not None:
                results.append(coerced)
    except Exception:
        return []
    return results


def _run_ddg_search(query: str):
    if _ddg_search is None:
        return []
    return _ddg_search.invoke(query)


def search(query: str, limit: int | None = None) -> list[dict]:
    settings = get_settings()
    limit = limit or settings.search_result_limit
    normalized_query = _normalize_query(query)
    cache_key = f"search:{normalized_query}:{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    raw_results = _collect_firecrawl_results(query, limit) + _collect_ddg_results(query)

    deduped_results: list[dict] = []
    seen_urls = set()
    for item in sorted(raw_results, key=lambda result: result.get("relevance_score", 0), reverse=True):
        url = item.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_results.append(item)
        if len(deduped_results) >= limit:
            break

    _cache_set(cache_key, deduped_results, settings.search_cache_ttl_seconds)
    return deduped_results


def scrape(url: str) -> dict:
    settings = get_settings()
    normalized_url = normalize_url(url)
    cache_key = f"scrape:{normalized_url}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    payload = {
        "url": normalized_url,
        "title": "",
        "content": "",
        "success": False,
    }

    if _app is not None:
        try:
            response = _app.scrape(normalized_url, formats=["markdown"])
            if response and response.markdown:
                payload = {
                    "url": normalized_url,
                    "title": response.metadata.title if response.metadata else "",
                    "content": response.markdown,
                    "success": True,
                }
        except Exception:
            payload = payload

    if not payload["success"] and trafilatura is not None:
        try:
            downloaded = trafilatura.fetch_url(normalized_url)
            if downloaded:
                content = trafilatura.extract(downloaded, include_links=True, include_images=False)
                if content:
                    payload = {
                        "url": normalized_url,
                        "title": "Extracted Content",
                        "content": content,
                        "success": True,
                    }
        except Exception:
            payload = payload

    _cache_set(cache_key, payload, settings.scrape_cache_ttl_seconds)
    return payload
