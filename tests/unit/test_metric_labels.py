"""Regression tests for lucy/metric_labels.py — the deterministic labeler
that replaced asking the model to classify high/low/moderate itself
(which was unreliable even when given the exact thresholds)."""

from lucy.metric_labels import label_fundamentals


def test_aapl_real_values_labeled_correctly():
    """Real AAPL fundamentals from 2026-07-17 testing."""
    fundamentals = {
        "pe": 40.404358, "forward_pe": 34.625034, "peg": 2.54, "div_yield": 0.32,
        "roe": 1.4147099, "pm": 0.27152002, "de": 79.548, "analyst_rating_mean": 2.04255,
    }
    labels = label_fundamentals(fundamentals)

    assert labels["pe_label"] == "high"
    assert labels["forward_pe_label"] == "high"
    assert labels["peg_label"] == "high/expensive"
    assert labels["div_yield_label"] == "low"
    assert labels["roe_label"] == "strong"
    assert labels["pm_label"] == "strong"
    assert labels["analyst_rating_mean_label"] == "Buy"


def test_debt_to_equity_moderate_not_high():
    """The exact regression this module was built to fix: the model kept
    calling AAPL's real 79.5 debt-to-equity "high" when it's actually
    "moderate" per the project's own reference ranges (50-150)."""
    labels = label_fundamentals({"de": 79.548})
    assert labels["de_label"] == "moderate"


def test_debt_to_equity_buckets():
    assert label_fundamentals({"de": 6.555})["de_label"] == "low/conservative leverage"
    assert label_fundamentals({"de": 49.9})["de_label"] == "low/conservative leverage"
    assert label_fundamentals({"de": 50.0})["de_label"] == "moderate"
    assert label_fundamentals({"de": 149.9})["de_label"] == "moderate"
    assert label_fundamentals({"de": 150.0})["de_label"] == "high leverage"
    assert label_fundamentals({"de": 500})["de_label"] == "high leverage"


def test_analyst_rating_scale_buckets():
    assert label_fundamentals({"analyst_rating_mean": 1.0})["analyst_rating_mean_label"] == "Strong Buy"
    assert label_fundamentals({"analyst_rating_mean": 1.29508})["analyst_rating_mean_label"] == "Strong Buy"
    assert label_fundamentals({"analyst_rating_mean": 2.04255})["analyst_rating_mean_label"] == "Buy"
    assert label_fundamentals({"analyst_rating_mean": 3.0})["analyst_rating_mean_label"] == "Hold"
    assert label_fundamentals({"analyst_rating_mean": 4.0})["analyst_rating_mean_label"] == "Sell"
    assert label_fundamentals({"analyst_rating_mean": 5.0})["analyst_rating_mean_label"] == "Strong Sell"


def test_missing_metrics_produce_no_label():
    labels = label_fundamentals({"pe": None, "de": None})
    assert "pe_label" not in labels
    assert "de_label" not in labels


def test_empty_fundamentals_produces_empty_labels():
    assert label_fundamentals({}) == {}


def test_roe_and_pm_use_percent_scale_not_raw_fraction():
    """roe/pm are stored as raw decimal fractions (0.66 = 66%) — the
    labeler must scale by 100 before bucketing against the percent
    thresholds, not bucket the raw fraction directly."""
    labels = label_fundamentals({"roe": 0.05, "pm": 0.03})
    assert labels["roe_label"] == "low"
    assert labels["pm_label"] == "low"
