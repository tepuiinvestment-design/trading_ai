from lucy.rag.news_verification import (
    check_price_contradiction,
    verify_article,
    verify_articles,
)


def _article(headline="", summary="", source="Reuters", url="https://www.reuters.com/markets/foo"):
    return {"headline": headline, "summary": summary, "source": source, "url": url, "published": "2026-09-18"}


def test_wire_source_is_verified():
    result = verify_article(_article(source="Reuters", url="https://www.reuters.com/markets/foo"))
    assert result["verdict"] == "verified"
    assert result["source_tier"] == "wire"
    assert result["confidence"] >= 0.9


def test_research_source_is_verified():
    """One of Lucy's own four research sources should verify cleanly."""
    result = verify_article(_article(source="Yahoo Finance", url="https://finance.yahoo.com/news/foo"))
    assert result["verdict"] == "verified"
    assert result["source_tier"] == "aggregator"


def test_unknown_source_is_unverified_not_fake():
    """An unrecognized source is 'unverified', never a blanket 'fake' —
    that's the only claim this module can actually back up."""
    result = verify_article(_article(source="Some Random Blog", url="https://some-random-blog.example/post"))
    assert result["verdict"] == "unverified"
    assert result["source_tier"] == "unverified"
    assert result["confidence"] < 0.5
    assert any("registry" in r for r in result["reasons"])


def test_no_source_or_url_is_unverified():
    result = verify_article({"headline": "Something happened", "summary": "", "source": None, "url": None})
    assert result["verdict"] == "unverified"


def test_typosquat_domain_flagged_even_with_trusted_name():
    """A byline claiming Reuters whose link is actually a lookalike
    domain is the classic source-spoofing pattern — must be flagged
    even though the name alone would pass."""
    result = verify_article(_article(source="Reuters", url="https://www.reutors.com/markets/foo"))
    assert result["verdict"] == "unverified"
    assert any("typosquat" in r or "resembles" in r for r in result["reasons"])


def test_trusted_name_with_unrecognized_domain_is_suspicious():
    result = verify_article(_article(source="Bloomberg", url="https://totally-unrelated-site.example/story"))
    assert result["verdict"] == "unverified"
    assert any("isn't one Lucy recognizes" in r for r in result["reasons"])


def test_price_contradiction_flags_wildly_wrong_price():
    flagged = check_price_contradiction("Shares trading at $9.50 a share today", current_price=185.0)
    assert flagged == [9.50]


def test_price_contradiction_allows_normal_drift():
    """A ~5% difference (normal intraday movement/staleness) must not
    trip the check — only a wild, implausible figure should."""
    flagged = check_price_contradiction("Shares trading at $180.00 a share today", current_price=185.0)
    assert flagged == []


def test_price_contradiction_ignores_unrelated_dollar_figures():
    """A dollar figure with no price-context keyword nearby (e.g. a
    revenue number) must not be treated as a price claim at all."""
    flagged = check_price_contradiction("The company reported $185 million in quarterly revenue", current_price=9.50)
    assert flagged == []


def test_price_contradiction_no_current_price_available():
    assert check_price_contradiction("Shares trading at $9.50 a share today", current_price=None) == []


def test_contradicted_price_overrides_trusted_source():
    article = _article(
        source="Reuters",
        url="https://www.reuters.com/markets/foo",
        headline="Stock hits $9.50 a share amid selloff",
    )
    result = verify_article(article, current_price=185.0)
    assert result["verdict"] == "contradicted"
    assert result["confidence"] <= 0.1


def test_corroboration_raises_confidence():
    baseline = verify_article(_article(source="MSN", url="https://www.msn.com/en-us/money/foo"))
    boosted = verify_article(
        _article(source="MSN", url="https://www.msn.com/en-us/money/foo"), corroborated_by=2
    )
    assert boosted["confidence"] > baseline["confidence"]
    assert any("corroborated" in r for r in boosted["reasons"])


def test_verify_articles_batch_adds_verification_and_corroborates():
    articles = [
        {"headline": "Acme Corp beats earnings expectations", "summary": "", "source": "Reuters",
         "url": "https://www.reuters.com/markets/acme"},
        {"headline": "Acme Corp beats earnings expectations this quarter", "summary": "", "source": "Yahoo Finance",
         "url": "https://finance.yahoo.com/news/acme"},
        {"headline": "Totally unrelated story about a different company", "summary": "", "source": "Some Blog",
         "url": "https://some-blog.example/post"},
    ]
    result = verify_articles(articles, current_price=None)

    assert all("verification" in a for a in result)
    assert result[0]["verification"]["verdict"] == "verified"
    # Corroborated by the near-identical Yahoo Finance headline.
    assert "corroborated" in " ".join(result[0]["verification"]["reasons"])
    assert result[2]["verification"]["verdict"] == "unverified"


def test_verify_articles_empty_list():
    assert verify_articles([]) == []


def test_verify_articles_same_source_does_not_self_corroborate():
    articles = [
        {"headline": "Acme Corp beats earnings expectations", "summary": "", "source": "Reuters",
         "url": "https://www.reuters.com/markets/acme-1"},
        {"headline": "Acme Corp beats earnings expectations", "summary": "", "source": "Reuters",
         "url": "https://www.reuters.com/markets/acme-2"},
    ]
    result = verify_articles(articles)
    assert "corroborated" not in " ".join(result[0]["verification"]["reasons"])
