import yfinance as yf
import requests
from bs4 import BeautifulSoup

def fetch_market_price(symbol: str):
    """
    Lucy market price fetcher using:
    1. Yahoo Finance (primary, stable, real-time)
    2. MSN Money (fallback HTML scraper)
    """

    symbol = symbol.upper()

    # ---------------------------------------------------------
    # 1) YAHOO FINANCE (PRIMARY)
    # ---------------------------------------------------------
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if info and "regularMarketPrice" in info:
            return {
                "status": "ok",
                "symbol": info.get("symbol", symbol),
                "price": info.get("regularMarketPrice"),
                "open": info.get("regularMarketOpen"),
                "high": info.get("regularMarketDayHigh"),
                "low": info.get("regularMarketDayLow"),
                "source": "Yahoo Finance"
            }
    except Exception:
        pass  # Yahoo failed → fallback to MSN Money

    # ---------------------------------------------------------
    # 2) MSN MONEY (FALLBACK)
    # ---------------------------------------------------------
    try:
        msn_url = f"https://www.msn.com/en-us/money/stockdetails/analysis/{symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        response = requests.get(msn_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return {
                "status": "error",
                "message": "MSN Money request failed."
            }

        soup = BeautifulSoup(response.text, "html.parser")

        # MSN Money price selector
        price_tag = soup.select_one("span.currentval")
        if not price_tag:
            return {
                "status": "error",
                "message": "MSN Money: price not found."
            }

        price = float(price_tag.text.replace(",", ""))

        # MSN does not always provide OHLC → return price only
        return {
            "status": "ok",
            "symbol": symbol,
            "price": price,
            "open": None,
            "high": None,
            "low": None,
            "source": "MSN Money"
        }

    except Exception:
        return {
            "status": "error",
            "message": "Both Yahoo Finance and MSN Money failed."
        }
