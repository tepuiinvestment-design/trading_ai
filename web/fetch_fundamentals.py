import yfinance as yf
import requests
from bs4 import BeautifulSoup

def fetch_fundamentals(symbol: str):
    """
    Lucy fundamentals fetcher using:
    1. Yahoo Finance (primary)
    2. MSN Money (fallback)
    """

    symbol = symbol.upper()

    # ---------------------------------------------------------
    # 1) YAHOO FINANCE (PRIMARY)
    # ---------------------------------------------------------
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if info:
            return {
                "status": "ok",
                "symbol": info.get("symbol", symbol),
                "pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg": info.get("pegRatio"),
                "div_yield": info.get("dividendYield"),
                "roe": info.get("returnOnEquity"),
                "pm": info.get("profitMargins"),
                "de": info.get("debtToEquity"),
                "market_cap": info.get("marketCap"),
                "eps_trailing": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "revenue_growth": info.get("revenueGrowth"),
                "free_cash_flow": info.get("freeCashflow"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "analyst_rating": info.get("recommendationKey"),
                "analyst_rating_mean": info.get("recommendationMean"),
                "analyst_count": info.get("numberOfAnalystOpinions"),
                "target_price_mean": info.get("targetMeanPrice"),
                "target_price_low": info.get("targetLowPrice"),
                "target_price_high": info.get("targetHighPrice"),
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

        def extract(label):
            tag = soup.find("span", string=label)
            if not tag:
                return None
            val_tag = tag.find_next("span")
            if not val_tag:
                return None
            try:
                return float(val_tag.text.replace(",", "").replace("%", ""))
            except:
                return val_tag.text

        return {
            "status": "ok",
            "symbol": symbol,
            "pe": extract("P/E Ratio"),
            "forward_pe": extract("Forward P/E"),
            "peg": extract("PEG Ratio"),
            "div_yield": extract("Dividend Yield"),
            "roe": extract("Return on Equity"),
            "pm": extract("Profit Margin"),
            "de": extract("Debt to Equity"),
            "market_cap": extract("Market Cap"),
            "eps_trailing": extract("EPS"),
            "revenue_growth": extract("Revenue Growth"),
            "source": "MSN Money"
        }

    except Exception:
        return {
            "status": "error",
            "message": "Both Yahoo Finance and MSN Money failed."
        }
