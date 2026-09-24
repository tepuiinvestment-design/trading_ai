def _bucket(value, thresholds_labels):
    """thresholds_labels: ordered [(upper_bound, label), ...]; the last
    entry's upper_bound should be None to catch everything above it."""
    if value is None:
        return None
    for upper, label in thresholds_labels:
        if upper is None or value < upper:
            return label
    return thresholds_labels[-1][1]


def label_fundamentals(fundamentals: dict) -> dict:
    """Pre-compute high/low/moderate labels for known metrics using the same
    thresholds previously left to the model to apply itself in the prompt
    (and observed to be unreliable — e.g. calling a 'moderate' 79.5
    debt-to-equity 'high'). The model's job becomes relaying a given label,
    not computing one from a raw number, which it does far more reliably."""
    labels = {}

    pe = fundamentals.get("pe")
    if pe is not None:
        labels["pe_label"] = _bucket(pe, [(15, "low"), (25, "moderate"), (None, "high")])

    forward_pe = fundamentals.get("forward_pe")
    if forward_pe is not None:
        labels["forward_pe_label"] = _bucket(forward_pe, [(15, "low"), (25, "moderate"), (None, "high")])

    peg = fundamentals.get("peg")
    if peg is not None:
        labels["peg_label"] = _bucket(
            peg, [(1, "undervalued relative to growth"), (2, "fairly valued"), (None, "high/expensive")]
        )

    div_yield = fundamentals.get("div_yield")
    if div_yield is not None:
        labels["div_yield_label"] = _bucket(div_yield, [(1, "low"), (3, "moderate"), (None, "high")])

    roe = fundamentals.get("roe")
    if roe is not None:
        labels["roe_label"] = _bucket(roe * 100, [(10, "low"), (20, "good"), (None, "strong")])

    pm = fundamentals.get("pm")
    if pm is not None:
        labels["pm_label"] = _bucket(pm * 100, [(5, "low"), (20, "moderate"), (None, "strong")])

    de = fundamentals.get("de")
    if de is not None:
        labels["de_label"] = _bucket(
            de, [(50, "low/conservative leverage"), (150, "moderate"), (None, "high leverage")]
        )

    analyst_mean = fundamentals.get("analyst_rating_mean")
    if analyst_mean is not None:
        labels["analyst_rating_mean_label"] = _bucket(
            analyst_mean,
            [(1.5, "Strong Buy"), (2.5, "Buy"), (3.5, "Hold"), (4.5, "Sell"), (None, "Strong Sell")],
        )

    return labels
