"""Fake-news / unverifiable-source scanning for Lucy's news pipeline.

Deterministic and rule-based, in the same spirit as lucy/grounding.py:
no second model call, just checks that either pass or don't, and an
honest confidence score rather than a bare true/false. Two independent
signals combine into a verdict for each article:

1. Source credibility (lucy.rag.source_registry) — is the publisher,
   by name and (when a link is present) by domain, one Lucy actually
   recognizes? A named "trusted" outlet whose link domain doesn't
   match it, or a domain that's a near-miss typosquat of a known one,
   is treated as more suspicious than an honestly-unknown source.
2. A narrow contradiction check: does the headline/summary cite a
   dollar price, in a stock-price context, that's wildly off from the
   market price Lucy fetched independently this run? This is
   deliberately narrow (wide tolerance, requires an explicit
   price-context keyword) rather than flagging every number in the
   article that doesn't happen to appear in Lucy's own fetched data —
   a real news article legitimately cites far more figures than Lucy's
   fundamentals/market/performance snapshot contains, and treating
   every one of those as a "contradiction" would drown the real
   signal in false positives.

Corroboration across the batch (the same story reported independently
by more than one recognized source) raises confidence. A lone report
from a source Lucy can't place is not, by itself, declared "fake" —
only "unable to verify", which is the honest claim this module can
actually back up; conflating the two would be its own kind of
misinformation.
"""
import difflib
import re

from lucy.rag.source_registry import classify_domain, classify_source_name, lookalike_domain

VERDICT_VERIFIED = "verified"
VERDICT_UNVERIFIED = "unverified"
VERDICT_CONTRADICTED = "contradicted"

_TIER_BASE_CONFIDENCE = {
    "wire": 0.9,
    "aggregator": 0.7,
    "unverified": 0.25,
}

# Two articles in the same batch are treated as independently reporting
# the same story when their headlines are at least this similar.
_SIMILAR_HEADLINE_THRESHOLD = 0.6

# A dollar figure is only checked against the fetched market price when
# it appears near one of these — otherwise it's just as likely to be a
# price target, an unrelated dollar amount, or a different metric.
_PRICE_CONTEXT_PATTERN = re.compile(
    r"(?:\$\s?\d[\d,]*(?:\.\d+)?)\s*(?:a|per)?\s*share"
    r"|(?:share|stock)s?\s+(?:trading|trade|hit|hits|close[ds]?|open(?:ed)?)"
    r"\s+at\s+\$\s?\d[\d,]*(?:\.\d+)?"
    r"|\$\s?\d[\d,]*(?:\.\d+)?\s*(?=.{0,20}\bshare)",
    re.IGNORECASE,
)
_DOLLAR_PATTERN = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")

# A cited price this far from the real one (as a fraction) is treated as
# a contradiction — wide enough that normal after-hours drift, a stale
# quote from earlier in the day, or rounding never trips it.
_PRICE_TOLERANCE_FRACTION = 0.35


def _effective_tier(article: dict) -> tuple:
    """Resolve the source tier for one article, checking the byline
    name and (when present) the link's domain against each other."""
    name_tier = classify_source_name(article.get("source"))
    url = article.get("url")
    domain_tier, domain = (None, "")
    if url:
        domain_tier, domain = classify_domain(url)

    reasons = []

    spoof_match = lookalike_domain(domain) if domain else None
    if spoof_match:
        reasons.append(
            f"link domain '{domain}' closely resembles the known outlet "
            f"'{spoof_match}' but is not it — possible spoofed/typosquat source"
        )
        return "unverified", reasons

    # A byline claiming a trusted outlet whose own link doesn't match any
    # domain Lucy recognizes for that outlet is more suspicious than
    # either signal taken alone.
    if domain_tier == "unverified" and name_tier in ("wire", "aggregator"):
        reasons.append(
            f"claims to be from '{article.get('source')}' but its link "
            f"domain ('{domain}') isn't one Lucy recognizes for that outlet"
        )
        return "unverified", reasons

    if domain_tier in ("wire", "aggregator"):
        return domain_tier, reasons

    return name_tier, reasons


def _headline_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def check_price_contradiction(text: str, current_price) -> list:
    """Return any dollar figures cited in a stock-price context in
    `text` that are far enough from `current_price` to be a likely
    contradiction. Deliberately conservative — see module docstring."""
    if not text or current_price in (None, 0):
        return []

    flagged = []
    for context_match in _PRICE_CONTEXT_PATTERN.finditer(text):
        dollar_match = _DOLLAR_PATTERN.search(context_match.group())
        if not dollar_match:
            continue
        cited = float(dollar_match.group(1).replace(",", ""))
        if abs(cited - current_price) / current_price > _PRICE_TOLERANCE_FRACTION:
            flagged.append(cited)

    return flagged


def verify_article(article: dict, current_price=None, corroborated_by: int = 0) -> dict:
    """Score a single news article dict (as returned by
    web.fetch_news.fetch_news) for source credibility and consistency
    with the market price Lucy fetched independently this run."""
    tier, reasons = _effective_tier(article)

    headline = article.get("headline") or ""
    summary = article.get("summary") or ""
    contradicted_prices = check_price_contradiction(f"{headline} {summary}", current_price)
    if contradicted_prices:
        reasons.append(
            f"cites a share price ({', '.join(f'${p:,.2f}' for p in contradicted_prices)}) "
            f"far from the ${current_price:,.2f} Lucy fetched independently this run"
        )

    if contradicted_prices:
        verdict = VERDICT_CONTRADICTED
        confidence = 0.1
    elif tier in ("wire", "aggregator"):
        verdict = VERDICT_VERIFIED
        confidence = _TIER_BASE_CONFIDENCE[tier]
    else:
        verdict = VERDICT_UNVERIFIED
        confidence = _TIER_BASE_CONFIDENCE["unverified"]
        if not reasons:
            reasons.append("publisher not in Lucy's trusted source registry")

    if corroborated_by > 0 and verdict != VERDICT_CONTRADICTED:
        confidence = min(0.98, confidence + 0.05 * corroborated_by)
        reasons.append(
            f"corroborated by {corroborated_by} other independently-sourced "
            f"report(s) in this batch"
        )

    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "source_tier": tier,
        "reasons": reasons,
    }


def verify_articles(articles: list, current_price=None) -> list:
    """Verify a batch of articles (as returned by
    web.fetch_news.fetch_news), adding a "verification" key to each in
    place and returning the same list."""
    if not articles:
        return articles

    # Corroboration pass: for each article, count how many *other*
    # articles in the batch report a near-identical headline from a
    # different named source. Two syndications of the same story under
    # different bylines both count; the same source repeated doesn't
    # corroborate itself.
    corroboration_counts = []
    for i, article in enumerate(articles):
        count = 0
        for j, other in enumerate(articles):
            if i == j or article.get("source") == other.get("source"):
                continue
            if _headline_similarity(article.get("headline"), other.get("headline")) >= _SIMILAR_HEADLINE_THRESHOLD:
                count += 1
        corroboration_counts.append(count)

    for article, corroborated_by in zip(articles, corroboration_counts):
        article["verification"] = verify_article(
            article, current_price=current_price, corroborated_by=corroborated_by
        )

    return articles
