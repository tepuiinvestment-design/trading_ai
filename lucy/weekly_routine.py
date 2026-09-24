import os
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("LUCY_RAG_PERSIST_DIR", str(PROJECT_ROOT / "data" / "vectorstore"))
os.environ.setdefault("LUCY_WATCHLIST_PATH", str(PROJECT_ROOT / "data" / "watchlist.csv"))

from lucy.watchlist import load_watchlist
from lucy.analyze import run_full_analysis
from lucy.filing_ingestion import run_filing_ingestion

REPORTS_DIR = PROJECT_ROOT / "reports"


def _format_text_report(date_str: str, results: list) -> str:
    lines = [f"Lucy Weekly Report — {date_str}", "=" * 40, ""]

    for result in results:
        symbol = result.get("symbol", "?")
        lines.append(f"### {symbol}")

        if result.get("status") != "ok":
            lines.append(f"  Skipped/error: {result.get('message')}")
            lines.append("")
            continue

        market = result.get("market", {})
        lines.append(f"  Price: {market.get('price')}")
        lines.append("")
        lines.append("  Analysis:")
        lines.append(f"  {result.get('analysis', '')}")
        lines.append("")

    return "\n".join(lines)


def run_weekly_routine():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_watchlist()
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Filings change quarterly/annually at most (see filing_ingestion.py's
    # own docstring), so this runs once per weekly batch rather than being
    # called from run_full_analysis on every request. Running it first means
    # freshly ingested filings are available to this same run's
    # retrieve_context() calls below. A filing-fetch failure (e.g. SEC EDGAR
    # hiccup) shouldn't block the rest of the weekly routine, so it's
    # isolated in its own try/except.
    try:
        filing_results = run_filing_ingestion()
    except Exception as exc:
        filing_results = [{"status": "error", "message": str(exc)}]

    results = []
    for symbol in tickers:
        try:
            result = run_full_analysis(symbol)
        except Exception as exc:
            result = {"status": "error", "symbol": symbol, "message": str(exc)}
        results.append(result)

    json_path = REPORTS_DIR / f"{date_str}.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    txt_path = REPORTS_DIR / f"{date_str}.txt"
    txt_path.write_text(_format_text_report(date_str, results), encoding="utf-8")

    return {
        "status": "ok",
        "date": date_str,
        "tickers_processed": len(results),
        "filings_ingested": filing_results,
        "json_report": str(json_path),
        "txt_report": str(txt_path)
    }


if __name__ == "__main__":
    summary = run_weekly_routine()
    print(json.dumps(summary, indent=2))
