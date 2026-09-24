import os
from pathlib import Path


def _watchlist_path() -> Path:
    return Path(os.environ.get("LUCY_WATCHLIST_PATH", "data/watchlist.csv"))


def load_watchlist() -> list:
    path = _watchlist_path()
    if not path.exists():
        return []

    tickers = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ticker = line.upper()
        if ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def is_watched(symbol: str) -> bool:
    return symbol.upper() in load_watchlist()


def add_ticker(symbol: str) -> list:
    path = _watchlist_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    tickers = load_watchlist()
    symbol = symbol.upper()
    if symbol not in tickers:
        tickers.append(symbol)
        path.write_text("\n".join(tickers) + "\n", encoding="utf-8")
    return tickers


def remove_ticker(symbol: str) -> list:
    path = _watchlist_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    tickers = [t for t in load_watchlist() if t != symbol.upper()]
    path.write_text("\n".join(tickers) + ("\n" if tickers else ""), encoding="utf-8")
    return tickers
