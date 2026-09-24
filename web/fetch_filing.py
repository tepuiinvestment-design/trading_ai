import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# SEC EDGAR's fair-access policy requires every request to declare a real
# contact in the User-Agent, or requests can be rate-limited/blocked.
SEC_USER_AGENT = "Tepui tepuiinvestment@outlook.com"
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT}

_TICKER_CIK_CACHE_PATH = Path("data/sec_company_tickers.json")
_TICKER_CIK_CACHE_MAX_AGE_DAYS = 7
_TICKER_CIK_MAP = None

FORM_TYPES = ("10-K", "10-Q")
MAX_FILING_CHARS = 50000


def _load_ticker_cik_map() -> dict:
    """SEC's ticker->CIK mapping, cached locally since it rarely changes."""
    global _TICKER_CIK_MAP
    if _TICKER_CIK_MAP is not None:
        return _TICKER_CIK_MAP

    if _TICKER_CIK_CACHE_PATH.exists():
        age_days = (time.time() - _TICKER_CIK_CACHE_PATH.stat().st_mtime) / 86400
        if age_days < _TICKER_CIK_CACHE_MAX_AGE_DAYS:
            data = json.loads(_TICKER_CIK_CACHE_PATH.read_text(encoding="utf-8"))
            _TICKER_CIK_MAP = {v["ticker"].upper(): str(v["cik_str"]) for v in data.values()}
            return _TICKER_CIK_MAP

    response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    _TICKER_CIK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TICKER_CIK_CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")

    _TICKER_CIK_MAP = {v["ticker"].upper(): str(v["cik_str"]) for v in data.values()}
    return _TICKER_CIK_MAP


def _get_cik(symbol: str):
    return _load_ticker_cik_map().get(symbol.upper())


def _latest_filing_meta(cik: str):
    padded_cik = cik.zfill(10)
    response = requests.get(
        f"https://data.sec.gov/submissions/CIK{padded_cik}.json",
        headers=SEC_HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    for i, form in enumerate(forms):
        if form in FORM_TYPES:
            return {
                "form": form,
                "accession_number": accession_numbers[i],
                "primary_document": primary_docs[i],
                "filing_date": filing_dates[i],
            }
    return None


def _fetch_filing_text(cik: str, accession_number: str, primary_document: str) -> str:
    accession_no_dashes = accession_number.replace("-", "")
    cik_no_leading_zeros = str(int(cik))
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_leading_zeros}/{accession_no_dashes}/{primary_document}"
    )

    response = requests.get(url, headers=SEC_HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Modern filings are Inline XBRL — a hidden block of raw XBRL tags
    # (namespace URIs, context IDs) sits at the top of <body>. Strip it and
    # script/style tags, or get_text() returns machine metadata instead of
    # the actual filing prose.
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.IGNORECASE)):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"ix:header", re.IGNORECASE)):
        tag.decompose()
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    return text[:MAX_FILING_CHARS]


def fetch_latest_filing(symbol: str):
    symbol = symbol.upper()

    try:
        cik = _get_cik(symbol)
        if not cik:
            return {
                "status": "error",
                "symbol": symbol,
                "message": "No SEC CIK found for this ticker (may not be a "
                           "US-listed company with standard EDGAR filings)."
            }

        meta = _latest_filing_meta(cik)
        if not meta:
            return {
                "status": "error",
                "symbol": symbol,
                "message": "No recent 10-K/10-Q filing found for this ticker."
            }

        text = _fetch_filing_text(cik, meta["accession_number"], meta["primary_document"])
        if not text:
            return {
                "status": "error",
                "symbol": symbol,
                "message": "Filing document was empty after fetching."
            }

        return {
            "status": "ok",
            "symbol": symbol,
            "form": meta["form"],
            "filing_date": meta["filing_date"],
            "accession_number": meta["accession_number"],
            "text": text
        }

    except Exception as exc:
        return {
            "status": "error",
            "symbol": symbol,
            "message": str(exc)
        }
