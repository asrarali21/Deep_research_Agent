from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


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
    "template",
    "templates",
    "examples",
    "resource hub",
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


_TRACKING_QUERY_PREFIXES = (
    "utm_",
    "gclid",
    "fbclid",
    "yclid",
    "mc_cid",
    "mc_eid",
    "mkt_tok",
    "ref",
    "ref_",
)


def normalize_url_for_matching(url: str) -> str:
    """Normalize URLs for equivalence checks (more aggressive than `normalize_url`).

    - Lower-case scheme/host
    - Remove fragment
    - Remove trailing slash (except root)
    - Drop common tracking query params (utm_*, gclid, fbclid, etc.)
    - Sort remaining query params for stable comparison
    """
    raw = (url or "").strip()
    if not raw:
        return ""

    split = urlsplit(raw)
    scheme = (split.scheme or "https").lower()
    netloc = split.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = split.path.rstrip("/") or "/"

    kept: list[tuple[str, str]] = []
    for key, value in parse_qsl(split.query, keep_blank_values=True):
        k = (key or "").strip()
        if not k:
            continue
        lowered = k.lower()
        if lowered.startswith("utm_"):
            continue
        if any(lowered == prefix or lowered.startswith(prefix) for prefix in _TRACKING_QUERY_PREFIXES):
            continue
        kept.append((k, value))
    kept.sort(key=lambda kv: (kv[0].lower(), kv[1]))
    query = urlencode(kept, doseq=True)

    return urlunsplit((scheme, netloc, path, query, ""))


def match_scraped_source(card_url: str, scraped_urls: list[str]) -> tuple[bool, str]:
    """Return (matched, best_url) where best_url is a representative scraped URL.

    Matching is done on `normalize_url_for_matching()`; when matched, we return
    the first scraped URL with the same normalized key (stable representative).
    """
    target = normalize_url_for_matching(card_url)
    if not target:
        return False, ""
    normalized_to_original: dict[str, str] = {}
    for src in scraped_urls:
        key = normalize_url_for_matching(src)
        if key and key not in normalized_to_original:
            normalized_to_original[key] = src
    best = normalized_to_original.get(target, "")
    return (bool(best), best)


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


# Important short tokens that should NOT be filtered out (domain-specific acronyms)
IMPORTANT_SHORT_TOKENS = {
    "ai", "ml", "ev", "agi", "nlp", "llm", "gpu", "cpu", "iot",
    "api", "5g", "6g", "ar", "vr", "xr", "eu", "us", "uk",
}


def stem_token(token: str) -> str:
    """Very lightweight suffix stripping for English.
    
    Not a full stemmer — just handles the most common inflections that cause
    false negatives in token matching (e.g. 'predictions' → 'predict',
    'timelines' → 'timeline', 'achieved' → 'achiev').
    """
    if len(token) <= 4:
        return token
    for suffix in ("ations", "ation", "ments", "ment", "ings", "ness", "tion", "sion",
                   "ence", "ance", "able", "ible", "ical", "ally", "ious", "eous",
                   "ives", "ful", "ous", "ing", "ies", "ers", "est", "ity",
                   "ive", "ent", "ant", "ial", "ed", "ly", "es", "er", "al", "ty"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def query_tokens(query: str) -> list[str]:
    """Extract meaningful tokens from a query string."""
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    result = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token in IMPORTANT_SHORT_TOKENS:
            result.append(token)
        elif len(token) > 2:
            result.append(token)
    return result


def content_tokens(text: str) -> set[str]:
    """Extract tokens from content text with stemming for fuzzy matching."""
    raw = query_tokens(text)
    stemmed = set()
    for token in raw:
        stemmed.add(token)
        stemmed.add(stem_token(token))
    return stemmed


def fuzzy_token_overlap(query_toks: list[str] | set[str], text: str) -> int:
    """Count how many query tokens appear in text via fuzzy matching.
    
    Uses multiple strategies to handle natural language inflections:
    1. Exact token match (query token in text's content tokens)
    2. Stemmed match (stem of query token in text's content tokens)
    3. Substring containment (query token found as substring in full text)
    4. Prefix matching (stems share a common prefix of 4+ chars)
    
    This handles 'predictions' matching 'predict', 'artificial' matching
    'artificially', 'timeline' matching 'timelines', etc.
    """
    text_lower = text.lower()
    text_token_set = content_tokens(text)
    hits = 0
    for token in query_toks:
        stemmed = stem_token(token)
        # Strategy 1 & 2: exact or stemmed token in text tokens
        if token in text_token_set or stemmed in text_token_set:
            hits += 1
            continue
        # Strategy 3: substring containment in full text
        if token in text_lower or stemmed in text_lower:
            hits += 1
            continue
        # Strategy 4: prefix matching between stems (handles predict/prediction/predictions)
        # Check if any text token shares a common stem prefix of 4+ chars
        min_prefix = min(len(stemmed), 4)
        if len(stemmed) >= min_prefix:
            stem_prefix = stemmed[:min_prefix]
            if any(
                text_tok.startswith(stem_prefix) or stemmed.startswith(text_tok[:min_prefix])
                for text_tok in text_token_set
                if len(text_tok) >= min_prefix
            ):
                hits += 1
    return hits


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


def compute_query_relevance(query: str, title: str, snippet: str, url: str) -> float:
    """Compute a 0.0-1.0 ratio of how many query tokens appear in the result.
    
    Uses fuzzy token matching (substring containment + basic stemming) to avoid
    false negatives from inflection differences.
    """
    tokens = query_tokens(query)
    if not tokens:
        return 1.0  # No meaningful tokens to compare — pass through
    text = f"{title} {snippet} {url}"
    hits = fuzzy_token_overlap(tokens, text)
    return hits / len(tokens)


def is_off_topic_result(query: str, title: str, snippet: str, url: str) -> bool:
    """Flag results where fewer than 15% of query tokens appear — likely off-topic."""
    relevance = compute_query_relevance(query, title, snippet, url)
    return relevance < 0.15


def score_search_result(query: str, title: str, snippet: str, url: str) -> float:
    text = f"{title} {snippet} {url}".lower()
    tokens = query_tokens(query)
    token_hits = fuzzy_token_overlap(tokens, text)
    source_type = infer_source_type(url)
    authority = compute_authority_score(url, source_type)

    # Content relevance dominates over domain authority
    score = float(authority) * 1.2 + float(token_hits) * 3.5

    # Zero-overlap penalty: results with no query token matches are almost certainly off-topic
    if tokens and token_hits == 0:
        score -= 20.0

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

    return score


def is_reference_usable(url: str) -> bool:
    if not url or is_blocked_reference_url(url):
        return False
    return not url.startswith("mailto:")
