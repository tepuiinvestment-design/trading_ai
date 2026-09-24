import time

from web.fetch_filing import fetch_latest_filing
from lucy.watchlist import load_watchlist
from lucy.rag import ingest

# Be polite to SEC EDGAR — each ticker costs 2 requests (submissions JSON +
# document fetch); a small delay avoids hammering it across a whole watchlist.
REQUEST_DELAY_SECONDS = 0.3


def run_filing_ingestion():
    """Fetch each watchlist ticker's latest 10-K/10-Q and ingest it into the
    RAG store. Filings change quarterly/annually at most, so this is meant
    to be run periodically on its own — not on every analysis call."""
    tickers = load_watchlist()
    results = []

    for symbol in tickers:
        filing = fetch_latest_filing(symbol)

        if filing.get("status") == "ok":
            source = f"{symbol}_filing_{filing['form']}_{filing['filing_date']}"
            ingest.ingest_filing(
                ticker=symbol,
                source=source,
                text=filing["text"],
                extra_metadata={
                    "filing_type": filing["form"],
                    "filed_date": filing["filing_date"],
                },
            )

        results.append({
            "symbol": symbol,
            "status": filing.get("status"),
            "form": filing.get("form"),
            "filing_date": filing.get("filing_date"),
            "message": filing.get("message"),
        })

        time.sleep(REQUEST_DELAY_SECONDS)

    return results


if __name__ == "__main__":
    import json
    summary = run_filing_ingestion()
    print(json.dumps(summary, indent=2))
