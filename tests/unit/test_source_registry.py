from lucy.rag.source_registry import (
    classify_domain,
    classify_source_name,
    is_trusted_source_name,
    lookalike_domain,
)


def test_wire_sources_classified_as_wire():
    for name in ["Reuters", "Bloomberg", "Associated Press", "PR Newswire"]:
        assert classify_source_name(name) == "wire"


def test_research_sources_classified_as_aggregator():
    """The four sites Lucy actually uses as research sources."""
    for name in ["Yahoo Finance", "Google Finance", "MSN Money", "StockAnalysis.com"]:
        assert classify_source_name(name) == "aggregator"


def test_existing_trusted_names_still_trusted():
    """web/fetch_news.py's original TRUSTED_SOURCES set must still all
    pass — this registry replaces that tuple, not its behavior."""
    for name in ["Yahoo Finance", "Yahoo Finance Video", "MarketWatch", "Reuters", "Barrons.com", "Bloomberg", "MSN"]:
        assert is_trusted_source_name(name) is True


def test_unknown_sources_unverified():
    for name in ["Motley Fool", "GuruFocus.com", "Trefis", "The Wall Street Journal", "AFP", ""]:
        assert classify_source_name(name) == "unverified"
        assert is_trusted_source_name(name) is False


def test_none_source_name_unverified():
    assert classify_source_name(None) == "unverified"
    assert is_trusted_source_name(None) is False


def test_classify_domain_recognizes_research_sources():
    assert classify_domain("https://finance.yahoo.com/news/foo")[0] == "aggregator"
    assert classify_domain("https://www.google.com/finance/quote/AAPL")[0] == "aggregator"
    assert classify_domain("https://www.msn.com/en-us/money/stockdetails/foo")[0] == "aggregator"
    assert classify_domain("https://stockanalysis.com/stocks/aapl/")[0] == "aggregator"


def test_classify_domain_recognizes_wire_domains():
    assert classify_domain("https://www.reuters.com/markets/foo")[0] == "wire"
    assert classify_domain("https://www.sec.gov/cgi-bin/browse-edgar")[0] == "wire"


def test_classify_domain_unknown():
    tier, domain = classify_domain("https://some-random-blog.example/post")
    assert tier == "unverified"
    assert domain == "some-random-blog.example"


def test_classify_domain_no_url():
    tier, domain = classify_domain(None)
    assert tier is None
    assert domain == ""


def test_lookalike_domain_catches_typosquat():
    assert lookalike_domain("reutors.com") == "reuters.com"
    assert lookalike_domain("reuters-news.com") == "reuters.com"


def test_lookalike_domain_does_not_flag_known_domains():
    assert lookalike_domain("reuters.com") is None
    assert lookalike_domain("finance.yahoo.com") is None


def test_lookalike_domain_does_not_flag_unrelated_domains():
    assert lookalike_domain("some-random-blog.example") is None
