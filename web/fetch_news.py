import re
import yfinance as yf

from lucy.rag.source_registry import is_trusted_source_name

# Corporate suffixes stripped when deriving a matchable name from yfinance's
# shortName (e.g. "Western Digital Corp" -> "Western Digital").
_CORP_SUFFIXES = re.compile(
    r"\b(inc\.?|incorporated|corp\.?|corporation|co\.?|company|ltd\.?|"
    r"limited|plc|llc|class\s+[a-z])\b",
    re.IGNORECASE,
)


def _is_trusted(source_name: str) -> bool:
    # Delegates to the shared registry (lucy/rag/source_registry.py) so
    # this filter and the news-verification module never drift apart on
    # what counts as a trusted publisher.
    return is_trusted_source_name(source_name)


def _relevance_tokens(symbol: str, short_name: str) -> set:
    """Tokens that would identify this specific company in a headline —
    yfinance's news feed returns loosely-related market news, not strictly
    ticker-matched articles (e.g. a Micron or Oracle story surfacing under
    a GOOGL query), so headlines/summaries are checked against these."""
    tokens = {symbol.lower()}
    if short_name:
        cleaned = _CORP_SUFFIXES.sub("", short_name).strip()
        if cleaned:
            tokens.add(cleaned.lower())
            first_word = cleaned.split()[0].lower()
            if len(first_word) > 2:
                tokens.add(first_word)
    return tokens


def _is_relevant(headline: str, summary: str, tokens: set) -> bool:
    text = f"{headline or ''} {summary or ''}".lower()
    return any(tok in text for tok in tokens)


def fetch_news(symbol: str, limit: int = 5):
    symbol = symbol.upper()

    try:
        ticker = yf.Ticker(symbol)
        raw_items = ticker.news or []

        short_name = None
        try:
            short_name = ticker.info.get("shortName")
        except Exception:
            pass
        tokens = _relevance_tokens(symbol, short_name)

        articles = []
        for item in raw_items:
            content = item.get("content", {})
            provider = content.get("provider") or {}
            canonical = content.get("canonicalUrl") or {}
            source_name = provider.get("displayName", "")
            headline = content.get("title")
            summary = content.get("summary")

            if content.get("contentType") == "VIDEO":
                continue
            if not _is_trusted(source_name):
                continue
            if not _is_relevant(headline, summary, tokens):
                continue

            articles.append({
                "headline": headline,
                "summary": summary,
                "source": source_name,
                "published": content.get("pubDate"),
                "url": canonical.get("url"),
            })

            if len(articles) >= limit:
                break

        return {
            "status": "ok",
            "symbol": symbol,
            "news": articles
        }

    except Exception as exc:
        return {
            "status": "error",
            "symbol": symbol,
            "message": str(exc)
        }
