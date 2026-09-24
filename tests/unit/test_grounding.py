"""Regression tests for lucy/grounding.py, formalizing the known-good and
known-bad cases found manually during development (see memory:
project-rag-grounding-status for the full incident history)."""

from lucy.grounding import check_grounding, check_numbers, check_named_products


MSFT_SOURCE = {
    "market": {"status": "ok", "symbol": "MSFT", "price": 401.1, "open": 398.31, "high": 405.99, "low": 392.05},
    "fundamentals": {
        "status": "ok", "symbol": "MSFT", "pe": 23.92, "forward_pe": 20.70, "peg": 1.19,
        "div_yield": 0.92, "roe": 0.34, "pm": 0.39, "de": 30.27,
    },
    "sector": {"status": "ok", "symbol": "MSFT", "sector": "Technology", "industry": "Software - Infrastructure"},
    "news": {"status": "ok", "symbol": "MSFT", "news": []},
}


def test_catches_fabricated_company_product_meta_cloud():
    """The original fabrication that started this whole investigation:
    the model invented a nonexistent "Meta Cloud" product."""
    text = "Microsoft competes with Meta Cloud in the enterprise space."
    flagged = check_named_products(text, str(MSFT_SOURCE))
    assert "Meta Cloud" in flagged


def test_catches_fabricated_company_product_facebook_cloud():
    """Same fabrication pattern, reworded — confirms the check isn't
    keyed to one exact phrase."""
    text = "Competitors include Amazon (AWS) and Meta (Facebook Cloud)."
    flagged = check_named_products(text, str(MSFT_SOURCE))
    assert "Facebook Cloud" in flagged


def test_known_real_products_not_flagged():
    text = "Microsoft competes with Amazon Web Services and Google Cloud."
    flagged = check_named_products(text, str(MSFT_SOURCE))
    assert flagged == []


def test_azure_ai_not_flagged_as_fabricated():
    """Azure AI is a real Microsoft product, not a suspicious invented
    '{Company} {Suffix}' pattern."""
    text = "Microsoft continues to invest heavily in Azure AI."
    flagged = check_named_products(text, str(MSFT_SOURCE))
    assert flagged == []


def test_sentence_starter_not_flagged_as_company_name():
    """'As AI continues to...' should not be read as a company called 'As'."""
    text = "As AI continues to transform industries, growth accelerates."
    flagged = check_named_products(text, str(MSFT_SOURCE))
    assert flagged == []


def test_heading_gerund_not_flagged_as_company_name():
    """'Expanding AI and Cloud Offerings' is a section heading, not a
    company called 'Expanding'."""
    text = "Expanding AI and Cloud Offerings in Emerging Markets."
    flagged = check_named_products(text, str(MSFT_SOURCE))
    assert flagged == []


def test_clean_analysis_has_no_flags():
    text = (
        "- Key risks: Market volatility could impact price.\n"
        "- Key opportunities: Strong fundamentals support growth.\n"
        "- Final verdict: MSFT remains a solid long-term holding."
    )
    result = check_grounding(text, MSFT_SOURCE, str(MSFT_SOURCE))
    assert result["clean"] is True
    assert result["unverified_numbers"] == []
    assert result["unverified_products"] == []


def test_fabricated_number_is_flagged():
    text = "MSFT trades at a P/E of 87.3, well above historical norms."
    flagged = check_numbers(text, MSFT_SOURCE)
    assert "87.3" in flagged


def test_real_number_from_source_not_flagged():
    text = "MSFT trades at a P/E of 23.92."
    flagged = check_numbers(text, MSFT_SOURCE)
    assert flagged == []


def test_percent_claim_matches_raw_decimal_fraction():
    """yfinance stores ROE/profit-margin as raw decimal fractions (0.34),
    but the model correctly expresses them as percentages (34%) — the
    checker must compare against value/100, not just the raw value."""
    text = "MSFT has a strong ROE of 34%."
    flagged = check_numbers(text, MSFT_SOURCE)
    assert flagged == []


def test_sp500_index_reference_not_flagged():
    """'S&P 500' should never be read as citing the number 500."""
    text = "MSFT is a component of the S&P 500 index."
    flagged = check_numbers(text, MSFT_SOURCE)
    assert flagged == []


def test_52_week_phrase_not_flagged():
    """'52-week high/low' is a common phrase, not a citation of the
    number 52."""
    text = "The stock is trading near its 52-week high."
    flagged = check_numbers(text, MSFT_SOURCE)
    assert flagged == []


def test_calendar_year_not_flagged():
    text = "### MSFT Market Analysis (July 16, 2026)"
    flagged = check_numbers(text, MSFT_SOURCE)
    assert "2026" not in flagged


def test_list_numbering_not_flagged():
    """Bullet/section numbering like '2.' should never be read as a
    citable decimal number."""
    text = "1. Market Volatility\n2. Regulatory Uncertainty\n3. Competition"
    flagged = check_numbers(text, MSFT_SOURCE)
    assert flagged == []


def test_units_magnitude_error_is_caught():
    """Real regression: the model stated AAPL's free cash flow as
    "$101 trillion" when the real value is ~$101 billion
    (101,090,746,368) — a 1000x units error. The bare '101' shouldn't
    match the real ~101-billion figure at any reasonable tolerance."""
    source = {"fundamentals": {"free_cash_flow": 101090746368}}
    text = "The company has robust free cash flow ($101 trillion)."
    flagged = check_numbers(text, source)
    assert "101" in flagged
