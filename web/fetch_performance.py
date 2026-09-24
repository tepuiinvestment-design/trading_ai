from datetime import datetime, timedelta

import yfinance as yf


def _pct_change(start_price, end_price):
    if not start_price or end_price is None:
        return None
    return round((end_price - start_price) / start_price * 100, 2)


def _closest_price(history, target_date):
    """Close price on the trading day closest to (but not after) target_date."""
    subset = history[history.index <= target_date]
    if subset.empty:
        return None
    return float(subset["Close"].iloc[-1])


def _return_since(history, latest_price, lookback_date, earliest_available):
    if lookback_date < earliest_available:
        return None  # not enough history — report as unavailable, don't guess
    start_price = _closest_price(history, lookback_date)
    return _pct_change(start_price, latest_price)


def fetch_performance(symbol: str):
    symbol = symbol.upper()

    try:
        ticker = yf.Ticker(symbol)
        # Fetch more than 5y of headroom — an exact "5y" window puts the
        # 5-years-ago lookback right at the edge of available data, which
        # can spuriously fail depending on leap years/trading-day rounding.
        history = ticker.history(period="10y")

        if history.empty:
            return {
                "status": "error",
                "symbol": symbol,
                "message": "No historical price data available."
            }

        latest_price = float(history["Close"].iloc[-1])
        today = history.index[-1]
        earliest_available = history.index[0]

        one_month_ago = today - timedelta(days=30)
        ytd_start = datetime(today.year, 1, 1, tzinfo=today.tzinfo)
        one_year_ago = today - timedelta(days=365)
        three_years_ago = today - timedelta(days=365 * 3)
        five_years_ago = today - timedelta(days=365 * 5)

        return {
            "status": "ok",
            "symbol": symbol,
            "current_price": round(latest_price, 2),
            "return_1mo_pct": _return_since(history, latest_price, one_month_ago, earliest_available),
            "return_ytd_pct": _return_since(history, latest_price, ytd_start, earliest_available),
            "return_1yr_pct": _return_since(history, latest_price, one_year_ago, earliest_available),
            "return_3yr_pct": _return_since(history, latest_price, three_years_ago, earliest_available),
            "return_5yr_pct": _return_since(history, latest_price, five_years_ago, earliest_available),
            "history_start_date": str(earliest_available.date()),
            "source": "Yahoo Finance"
        }

    except Exception as exc:
        return {
            "status": "error",
            "symbol": symbol,
            "message": str(exc)
        }
