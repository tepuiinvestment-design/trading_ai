"""Publisher/domain credibility registry shared across Lucy's news
pipeline (web/fetch_news.py) and the news-verification module
(lucy/rag/news_verification.py) — a single source of truth so the two
never drift out of sync.

Tiers:
- "wire": the original reporting outlet for financial/market news —
  wire services, regulators, and official filing distributors. Highest
  trust; these are primary sources.
- "aggregator": the four sites Lucy actually uses as day-to-day
  research sources (Yahoo Finance, Google Finance, MSN Money,
  stockanalysis.com), plus the other outlets already treated as
  trusted by web/fetch_news.py (MarketWatch, Barron's). Legitimate and
  trusted enough to act on, but they carry/aggregate reporting rather
  than always being the original source of it.
- "unverified": anything not in this registry. This is not a claim
  that the publisher is fake — only that Lucy has no independent way
  to confirm who is actually behind it, which is exactly the gap this
  module exists to flag.
"""
import difflib
from urllib.parse import urlparse

# Original wire services / regulators / filing distributors.
WIRE_SOURCES = {
    "reuters",
    "bloomberg",
    "associated press",
    "ap news",
    "dow jones",
    "dow jones newswires",
    "sec edgar",
    "sec.gov",
    "pr newswire",
    "business wire",
    "globe newswire",
}

# Lucy's own research sources plus the other outlets web/fetch_news.py
# already treats as trusted.
AGGREGATOR_SOURCES = {
    "yahoo finance",
    "google finance",
    "msn",
    "msn money",
    "stockanalysis",
    "stockanalysis.com",
    "marketwatch",
    "barrons",
}

WIRE_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "apnews.com",
    "sec.gov",
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
}

# Lucy's four research sources (finance.yahoo.com, google.com/finance,
# msn.com/.../money, stockanalysis.com) plus MarketWatch/Barron's.
AGGREGATOR_DOMAINS = {
    "finance.yahoo.com",
    "yahoo.com",
    "google.com",
    "msn.com",
    "stockanalysis.com",
    "marketwatch.com",
    "barrons.com",
}

_ALL_KNOWN_DOMAINS = WIRE_DOMAINS | AGGREGATOR_DOMAINS


def _normalize_name(name: str) -> str:
    return (name or "").lower().strip()


def classify_source_name(name: str) -> str:
    """Classify a publisher/byline name (e.g. yfinance's
    provider.displayName) into "wire" / "aggregator" / "unverified"."""
    normalized = _normalize_name(name)
    if not normalized:
        return "unverified"
    if any(wire in normalized for wire in WIRE_SOURCES):
        return "wire"
    if any(agg in normalized for agg in AGGREGATOR_SOURCES):
        return "aggregator"
    return "unverified"


def is_trusted_source_name(name: str) -> bool:
    """Backward-compatible boolean check used by web/fetch_news.py —
    true for anything at aggregator tier or above."""
    return classify_source_name(name) in ("wire", "aggregator")


def _root_domain(url: str) -> str:
    if not url:
        return ""
    try:
        netloc = str(urlparse(url).netloc).lower()
    except Exception:
        return ""
    netloc = netloc.split("@")[-1].split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def classify_domain(url: str):
    """Classify a URL's root domain. Returns (tier, domain); tier is
    None (not "unverified") when no URL/domain could be parsed at all,
    so callers can distinguish "no link given" from "link given but
    not recognized"."""
    domain = _root_domain(url)
    if not domain:
        return None, ""
    if domain in WIRE_DOMAINS:
        return "wire", domain
    if domain in AGGREGATOR_DOMAINS:
        return "aggregator", domain
    return "unverified", domain


def lookalike_domain(domain: str, threshold: float = 0.8):
    """Detect a domain that closely resembles — but isn't — a known
    trusted domain, the classic typosquat/spoof pattern (e.g.
    "reuters-news.com" or "reutors.com" standing in for "reuters.com").
    Returns the trusted domain it resembles, or None if it isn't close
    to anything or already is one of the known domains."""
    if not domain or domain in _ALL_KNOWN_DOMAINS:
        return None

    best_match, best_ratio = None, 0.0
    for known in _ALL_KNOWN_DOMAINS:
        ratio = difflib.SequenceMatcher(None, domain, known).ratio()
        if ratio > best_ratio:
            best_match, best_ratio = known, ratio

    return best_match if best_ratio >= threshold else None
