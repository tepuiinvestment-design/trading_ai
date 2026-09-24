import pytest

from lucy import watchlist


@pytest.fixture
def temp_watchlist(tmp_path, monkeypatch):
    """Point LUCY_WATCHLIST_PATH at a scratch file for the duration of the
    test — never touches the real data/watchlist.csv."""
    path = tmp_path / "watchlist.csv"
    monkeypatch.setenv("LUCY_WATCHLIST_PATH", str(path))
    return path


def test_load_watchlist_missing_file_returns_empty(temp_watchlist):
    assert watchlist.load_watchlist() == []


def test_add_ticker_persists(temp_watchlist):
    watchlist.add_ticker("NVDA")
    assert watchlist.load_watchlist() == ["NVDA"]


def test_add_ticker_is_case_insensitive_and_deduped(temp_watchlist):
    watchlist.add_ticker("nvda")
    watchlist.add_ticker("NVDA")
    watchlist.add_ticker("Nvda")
    assert watchlist.load_watchlist() == ["NVDA"]


def test_add_multiple_tickers_preserves_order(temp_watchlist):
    for t in ["AAPL", "MSFT", "NVDA"]:
        watchlist.add_ticker(t)
    assert watchlist.load_watchlist() == ["AAPL", "MSFT", "NVDA"]


def test_remove_ticker(temp_watchlist):
    watchlist.add_ticker("AAPL")
    watchlist.add_ticker("MSFT")
    watchlist.remove_ticker("AAPL")
    assert watchlist.load_watchlist() == ["MSFT"]


def test_remove_nonexistent_ticker_is_a_no_op(temp_watchlist):
    watchlist.add_ticker("AAPL")
    watchlist.remove_ticker("ZZZZ")
    assert watchlist.load_watchlist() == ["AAPL"]


def test_is_watched(temp_watchlist):
    watchlist.add_ticker("AAPL")
    assert watchlist.is_watched("AAPL") is True
    assert watchlist.is_watched("aapl") is True
    assert watchlist.is_watched("MSFT") is False


def test_comment_lines_are_ignored(temp_watchlist):
    temp_watchlist.write_text("# comment\nAAPL\n# another\nMSFT\n", encoding="utf-8")
    assert watchlist.load_watchlist() == ["AAPL", "MSFT"]


def test_blank_lines_are_ignored(temp_watchlist):
    temp_watchlist.write_text("AAPL\n\n\nMSFT\n", encoding="utf-8")
    assert watchlist.load_watchlist() == ["AAPL", "MSFT"]
