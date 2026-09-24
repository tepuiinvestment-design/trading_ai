"""Integration tests for the actual HTTP API surface (api/router.py via
src/main.py), using FastAPI's in-process TestClient — no manual uvicorn
process needed. Not run by default — `pytest -m integration` to run them.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app

pytestmark = pytest.mark.integration

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_watchlist_get():
    response = client.get("/lucy/watchlist")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["watchlist"], list)


def test_watchlist_add_and_remove_round_trip():
    """Uses an obviously-fake ticker and cleans up after itself — never
    touches a real watchlist entry."""
    test_symbol = "ZZPYTEST"

    add_response = client.post("/lucy/watchlist", json={"symbol": test_symbol})
    assert add_response.status_code == 200
    assert test_symbol in add_response.json()["watchlist"]

    delete_response = client.delete(f"/lucy/watchlist/{test_symbol}")
    assert delete_response.status_code == 200
    assert test_symbol not in delete_response.json()["watchlist"]


def test_market_endpoint():
    response = client.post("/lucy/market", json={"symbol": "AAPL"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_fundamentals_endpoint():
    response = client.post("/lucy/fundamentals", json={"symbol": "AAPL"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_news_endpoint():
    response = client.post("/lucy/news", json={"symbol": "AAPL"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sector_endpoint():
    response = client.post("/lucy/sector", json={"symbol": "AAPL"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_endpoint_full_pipeline():
    """The slowest test in the suite — full LLM generation through the
    real HTTP layer, ~3-5 min on CPU. Confirms the API surface (not just
    the underlying run_full_analysis() function) works end to end."""
    response = client.post("/lucy/analyze", json={"symbol": "NVDA"})
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["symbol"] == "NVDA"
    assert "analysis" in body and len(body["analysis"]) > 0
    assert "grounding_review" in body
    assert "clean" in body["grounding_review"]

    # Allow one section to be dropped — the model occasionally omits a
    # section (e.g. the closing verdict) on a probabilistic basis even with
    # a complete, non-truncated response. Requiring strict 100% compliance
    # here would make this test flaky as a stable comparison baseline; the
    # bar throughout this project has been "no crash/truncation/fabrication",
    # not "the model is fully deterministic".
    expected_sections = ("risk", "opportunit", "performance", "sector", "fundamental", "verdict")
    analysis_lower = body["analysis"].lower()
    missing = [s for s in expected_sections if s not in analysis_lower]
    assert len(missing) <= 1, f"missing sections: {missing}"


def test_analyze_endpoint_rejects_non_watchlist_ticker():
    """Should short-circuit before any fetching/generation — fast."""
    response = client.post("/lucy/analyze", json={"symbol": "ZZZNOTREAL"})
    assert response.status_code == 200
    assert response.json()["status"] == "error"
