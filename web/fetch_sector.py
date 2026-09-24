import yfinance as yf

# Select Sector SPDR ETFs — a standard, liquid proxy for each GICS sector's
# recent performance.
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}


def fetch_sector(symbol: str):
    symbol = symbol.upper()

    try:
        info = yf.Ticker(symbol).info
        sector = info.get("sector")
        industry = info.get("industry")

        if not sector:
            return {
                "status": "error",
                "symbol": symbol,
                "message": "No sector data available for this symbol (may be an ETF/fund rather than a company)."
            }

        result = {
            "status": "ok",
            "symbol": symbol,
            "sector": sector,
            "industry": industry
        }

        etf_symbol = SECTOR_ETF_MAP.get(sector)
        if etf_symbol:
            etf_info = yf.Ticker(etf_symbol).info
            price = etf_info.get("regularMarketPrice")
            prev_close = etf_info.get("regularMarketPreviousClose")

            change_pct = None
            if price is not None and prev_close:
                change_pct = round((price - prev_close) / prev_close * 100, 2)

            result["sector_etf"] = {
                "symbol": etf_symbol,
                "price": price,
                "change_pct": change_pct
            }

        return result

    except Exception as exc:
        return {
            "status": "error",
            "symbol": symbol,
            "message": str(exc)
        }
