"""Tests for the pure filtering logic in web/fetch_news.py — no network
calls, just the trust/relevance/video-exclusion helpers."""

from web.fetch_news import _is_trusted, _relevance_tokens, _is_relevant


def test_trusted_sources_pass():
    for source in ["Yahoo Finance", "Yahoo Finance Video", "MarketWatch", "Reuters", "Barrons.com", "Bloomberg", "MSN"]:
        assert _is_trusted(source) is True


def test_untrusted_sources_rejected():
    for source in ["Motley Fool", "GuruFocus.com", "Trefis", "The Wall Street Journal", "AFP"]:
        assert _is_trusted(source) is False


def test_empty_source_rejected():
    assert _is_trusted("") is False
    assert _is_trusted(None) is False


def test_relevance_tokens_include_ticker_and_company_name():
    tokens = _relevance_tokens("WDC", "Western Digital Corp")
    assert "wdc" in tokens
    assert "western digital" in tokens
    assert "western" in tokens


def test_relevance_matches_ticker_symbol():
    tokens = _relevance_tokens("GOOGL", "Alphabet Inc.")
    assert _is_relevant("GOOGL shares rose today", "", tokens) is True


def test_relevance_matches_company_name_even_without_ticker():
    """Real regression: an article headlined about Nvidia/TSMC/ASML that
    never mentions the ticker "WDC" but does mention "Western Digital"
    by name in the body should still count as relevant."""
    tokens = _relevance_tokens("WDC", "Western Digital Corp")
    headline = "Beyond the Nvidia boom: portfolio manager likes AI stocks TSMC, ASML"
    summary = "...tepid about cloud storage companies such as Western Digital, saying..."
    assert _is_relevant(headline, summary, tokens) is True


def test_irrelevant_article_rejected():
    """Real regression: a GOOGL news query surfaced a genuine article
    about Micron and Oracle that had nothing to do with Google/Alphabet
    — the model faithfully cited it, producing an analysis substantively
    about the wrong company. This must be filtered before it reaches the
    model at all."""
    tokens = _relevance_tokens("GOOGL", "Alphabet Inc.")
    headline = "Here's What Can End Micron's Stock Pain"
    summary = "Micron stock is down 25% in the past month..."
    assert _is_relevant(headline, summary, tokens) is False
