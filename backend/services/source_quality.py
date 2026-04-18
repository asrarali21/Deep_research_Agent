from __future__ import annotations

import re
from urllib.parse import urlsplit


AUTHORITATIVE_DOMAINS = (
    ".gov",
    ".edu",
    "aha.org",
    "acc.org",
    "escardio.org",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "mayoclinic.org",
    "clevelandclinic.org",
    "iea.org",
    "niti.gov.in",
    "evyatra.in",
    "e-amrit.niti.gov.in",
)

BLOCKED_REFERENCE_HOSTS = {
    "duckduckgo.com",
    "www.duckduckgo.com",
    "google.com",
    "www.google.com",
    "bing.com",
    "www.bing.com",
    "search.yahoo.com",
    "youtube.com",
    "www.youtube.com",
    "linkedin.com",
    "www.linkedin.com",
    "facebook.com",
    "www.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "x.com",
    "twitter.com",
    "www.twitter.com",
    "wikipedia.org",
    "www.wikipedia.org",
}

LOW_VALUE_HOST_HINTS = (
    "grandviewresearch.com",
    "mordorintelligence.com",
    "investopedia.com",
    "wikipedia.org",
)

GENERIC_WRITING_PAGE_HINTS = (
    "how to write",
    "research paper",
    "research papers",
    "paperpal",
    "researcher resources",
    "writing guide",
    "introduction guide",
    "essay",
    "thesis statement",
    "citation style",
    "mla format",
    "apa format",
)

WRITING_QUERY_HINTS = {
    "write",
    "writing",
    "paper",
    "papers",
    "essay",
    "thesis",
    "abstract",
    "introduction",
    "citation",
    "citations",
    "literature",
    "review",
    "format",
    "guide",
}

PAYWALL_OR_BLOCK_HINTS = (
    "bloomberg.com",
    "ft.com",
    "wsj.com",
)

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "about",
    "that",
    "this",
    "their",
    "what",
    "which",
    "will",
    "would",
    "have",
    "after",
    "before",
    "your",
    "where",
    "when",
    "into",
    "than",
    "over",
    "under",
    "major",
    "current",
    "latest",
}


def normalize_url(url: str) -> str:
    split = urlsplit(url.strip())
    path = split.path.rstrip("/") or "/"
    return split._replace(scheme=split.scheme.lower(), netloc=split.netloc.lower(), path=path, fragment="").geturl()


def get_hostname(url: str) -> str:
    return urlsplit(url).netloc.lower().replace("www.", "")


def looks_like_homepage(url: str) -> bool:
    split = urlsplit(url)
    path = split.path.rstrip("/")
    return not path or path == ""


def is_blocked_reference_url(url: str) -> bool:
    hostname = get_hostname(url)
    return hostname in BLOCKED_REFERENCE_HOSTS


def is_low_value_reference_url(url: str) -> bool:
    hostname = get_hostname(url)
    return any(hint in hostname for hint in LOW_VALUE_HOST_HINTS)


def is_probably_paywalled(url: str) -> bool:
    hostname = get_hostname(url)
    return any(hint in hostname for hint in PAYWALL_OR_BLOCK_HINTS)


def infer_source_type(url: str) -> str:
    split = urlsplit(url)
    hostname = split.netloc.lower()
    path = split.path.lower()
    if not hostname:
        return "unknown"
    if hostname.endswith(".gov"):
        return "government"
    if hostname.endswith(".edu"):
        return "academic"
    if "pubmed" in hostname or "ncbi" in hostname or "nejm" in hostname or "thelancet" in hostname or "jamanetwork" in hostname:
        return "journal"
    if any(token in hostname for token in ("clinic", "hospital", "healthsystem", "medicine")):
        return "hospital"
    if any(token in hostname for token in ("heart.org", "acc.org", "escardio.org", "who.int", "nih.gov", "iea.org", "gov.in", "nic.in")):
        return "guideline"
    if any(token in path for token in ("guideline", "guidelines", "statement", "scientific-statement", "policy", "notification")):
        return "guideline"
    if any(token in hostname for token in ("news", "cnn", "forbes", "reuters")):
        return "news"
    if any(token in hostname for token in ("healthline", "webmd", "verywellhealth")):
        return "commercial_health"
    return "nonprofit" if hostname.endswith(".org") else "commercial"


def compute_authority_score(url: str, source_type: str | None = None) -> int:
    hostname = get_hostname(url)
    source_type = source_type or infer_source_type(url)
    if any(domain in hostname for domain in AUTHORITATIVE_DOMAINS):
        return 10
    if source_type in {"guideline", "government", "journal"}:
        return 9
    if source_type in {"academic", "hospital", "nonprofit"}:
        return 8
    if source_type == "news":
        return 5
    if source_type == "commercial_health":
        return 4
    if source_type == "commercial":
        return 3
    return 2


def query_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def is_writing_help_query(query: str) -> bool:
    lowered = query.lower()
    tokens = set(query_tokens(query))
    if "research paper" in lowered or "literature review" in lowered:
        return True
    return bool(tokens & WRITING_QUERY_HINTS)


def is_generic_low_signal_result(query: str, title: str, snippet: str, url: str) -> bool:
    if not query or is_writing_help_query(query):
        return False

    text = f"{title} {snippet} {url}".lower()
    hit_count = sum(1 for hint in GENERIC_WRITING_PAGE_HINTS if hint in text)
    if hit_count == 0:
        return False

    # Treat obvious writing-help pages as low signal for normal research queries.
    if any(
        phrase in text
        for phrase in (
            "how to write",
            "research paper",
            "introduction guide",
            "paperpal",
            "researcher resources",
        )
    ):
        return True

    return hit_count >= 2 and any(marker in text for marker in ("guide", "template", "examples", "writing"))


def score_search_result(query: str, title: str, snippet: str, url: str) -> float:
    text = f"{title} {snippet} {url}".lower()
    tokens = query_tokens(query)
    token_hits = sum(1 for token in tokens if token in text)
    source_type = infer_source_type(url)
    authority = compute_authority_score(url, source_type)

    score = float(authority) * 2.2 + float(token_hits) * 1.7
    if looks_like_homepage(url):
        score -= 3.0
    if is_low_value_reference_url(url):
        score -= 3.0
    if is_generic_low_signal_result(query, title, snippet, url):
        score -= 8.0
    if is_probably_paywalled(url):
        score -= 2.0
    if is_blocked_reference_url(url):
        score -= 100.0

    if "india" in text or ".in" in get_hostname(url):
        score += 1.0
    if any(token in text for token in ("2026", "2025", "2024", "forecast", "projection", "policy", "pricing", "market")):
        score += 0.7

    return score


def is_reference_usable(url: str) -> bool:
    if not url or is_blocked_reference_url(url):
        return False
    return not url.startswith("mailto:")
