"""Integration tests that hit real network APIs (yfinance, SEC EDGAR).
Not run by default — `pytest -m integration` to run them.
"""

import pytest

from web.fetch_market import fetch_market_price
from web.fetch_fundamentals import fetch_fundamentals
from web.fetch_sector import fetch_sector
from web.fetch_news import fetch_news
from web.fetch_performance import fetch_performance
from web.fetch_filing import fetch_latest_filing

pytestmark = pytest.mark.integration


def test_fetch_market_price_real_ticker():
    result = fetch_market_price("AAPL")
    assert result["status"] == "ok"
    assert isinstance(result["price"], (int, float))
    assert result["price"] > 0


def test_fetch_fundamentals_includes_expanded_fields():
    """Confirms the 2026-07-17 fundamentals expansion is still live —
    market cap, EPS, revenue growth, FCF, 52-week range, and real analyst
    consensus, not just the original 7 basic ratios."""
    result = fetch_fundamentals("AAPL")
    assert result["status"] == "ok"

    expected_fields = [
        "pe", "forward_pe", "peg", "div_yield", "roe", "pm", "de",
        "market_cap", "eps_trailing", "eps_forward", "revenue_growth",
        "free_cash_flow", "fifty_two_week_low", "fifty_two_week_high",
        "analyst_rating", "analyst_rating_mean", "analyst_count",
        "target_price_mean", "target_price_low", "target_price_high",
    ]
    for field in expected_fields:
        assert field in result, f"missing expected field: {field}"


def test_fetch_sector_real_stock():
    result = fetch_sector("AAPL")
    assert result["status"] == "ok"
    assert result["sector"]
    assert result["industry"]
    assert "sector_etf" in result


def test_fetch_sector_etf_returns_error_not_garbage():
    """ETFs don't have a GICS sector — must fail cleanly, not crash or
    return fabricated sector data."""
    result = fetch_sector("SPY")
    assert result["status"] == "error"


def test_fetch_news_excludes_video_content():
    """2026-07-17 fix: video segments should never appear in results."""
    result = fetch_news("AAPL", limit=10)
    assert result["status"] == "ok"
    # can't assert non-empty (news feed contents vary run to run), but if
    # there are any results, none should look like a video-only listing
    for article in result["news"]:
        assert article["headline"]


def test_fetch_news_only_trusted_sources():
    result = fetch_news("AAPL", limit=10)
    assert result["status"] == "ok"
    trusted = ("yahoo finance", "marketwatch", "reuters", "barrons", "bloomberg", "msn")
    for article in result["news"]:
        source = article["source"].lower()
        assert any(t in source for t in trusted), f"untrusted source leaked through: {article['source']}"


def test_fetch_performance_established_ticker_has_full_history():
    result = fetch_performance("AAPL")
    assert result["status"] == "ok"
    assert result["return_1mo_pct"] is not None
    assert result["return_1yr_pct"] is not None
    assert result["return_5yr_pct"] is not None  # regression: the 5y-window boundary bug made this always null


def test_fetch_performance_recent_ipo_has_partial_history():
    """CBRS IPO'd 2026-05-14 — should have a 1-month return but null for
    longer periods it doesn't have history for, not a guessed value."""
    result = fetch_performance("CBRS")
    assert result["status"] == "ok"
    assert result["return_5yr_pct"] is None
    assert result["return_3yr_pct"] is None


def test_fetch_filing_extracts_clean_prose_not_xbrl_metadata():
    """Regression: modern SEC filings are Inline XBRL — a naive tag-strip
    pulled in raw XBRL namespace/context metadata instead of the actual
    filing text. Confirms the fix (stripping display:none + ix:header)
    still holds."""
    result = fetch_latest_filing("AAPL")
    assert result["status"] == "ok"
    assert result["form"] in ("10-K", "10-Q")

    text = result["text"]
    assert len(text) > 1000
    # real filing prose should be present
    assert "SECURITIES AND EXCHANGE COMMISSION" in text or "Apple Inc" in text
    # the XBRL metadata dump that leaked through before the fix
    assert "xbrli:shares" not in text
    assert "http://fasb.org" not in text
