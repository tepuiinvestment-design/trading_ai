from fastapi import APIRouter
from pydantic import BaseModel

from models.phi3_generate import generate_api
from web.fetch_market import fetch_market_price
from web.fetch_fundamentals import fetch_fundamentals
from web.fetch_news import fetch_news
from web.fetch_sector import fetch_sector
from lucy.analyze import run_full_analysis
from lucy.watchlist import load_watchlist, add_ticker, remove_ticker
from lucy.rag.news_verification import verify_articles
from lucy.filing_ingestion import run_filing_ingestion

router = APIRouter(prefix="/lucy", tags=["Lucy"])

# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9

class MarketRequest(BaseModel):
    symbol: str

class FundamentalsRequest(BaseModel):
    symbol: str

class NewsRequest(BaseModel):
    symbol: str

class AnalyzeRequest(BaseModel):
    symbol: str

class WatchlistRequest(BaseModel):
    symbol: str


# ---------------------------------------------------------
# Lucy Endpoints
# ---------------------------------------------------------

@router.post("/generate")
def generate_text(req: GenerateRequest):
    output = generate_api(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p
    )
    return {
        "status": "ok",
        "lucy_response": output
    }


@router.post("/market")
def market_data(req: MarketRequest):
    data = fetch_market_price(req.symbol)
    return data


@router.post("/fundamentals")
def fundamentals(req: FundamentalsRequest):
    data = fetch_fundamentals(req.symbol)
    return data


@router.post("/news")
def news(req: NewsRequest):
    data = fetch_news(req.symbol)
    return data


@router.post("/sector")
def sector(req: FundamentalsRequest):
    data = fetch_sector(req.symbol)
    return data


@router.post("/news/verify")
def verify_news(req: NewsRequest):
    """Fetch news for a symbol and scan it for unverifiable/spoofed
    sources and share-price claims that contradict the current market
    data — the same check lucy/analyze.py runs automatically before an
    analysis, exposed standalone for spot-checking."""
    data = fetch_news(req.symbol)
    if data.get("status") != "ok":
        return data

    market = fetch_market_price(req.symbol)
    current_price = market.get("price") if market.get("status") == "ok" else None
    data["news"] = verify_articles(data["news"], current_price=current_price)
    return data


@router.post("/analyze")
def analyze(req: AnalyzeRequest):
    result = run_full_analysis(req.symbol)
    return result


@router.post("/filings/ingest")
def ingest_filings():
    """Fetch each watchlist ticker's latest 10-K/10-Q from SEC EDGAR and
    ingest it into the RAG store. Filings change quarterly/annually at most
    (see lucy/filing_ingestion.py), so this is meant for periodic or manual
    triggering — it also now runs automatically once per weekly routine
    batch (lucy/weekly_routine.py), not on every /lucy/analyze request."""
    results = run_filing_ingestion()
    return {
        "status": "ok",
        "results": results
    }


@router.get("/watchlist")
def get_watchlist():
    return {
        "status": "ok",
        "watchlist": load_watchlist()
    }


@router.post("/watchlist")
def post_watchlist(req: WatchlistRequest):
    return {
        "status": "ok",
        "watchlist": add_ticker(req.symbol)
    }


@router.delete("/watchlist/{symbol}")
def delete_watchlist(symbol: str):
    return {
        "status": "ok",
        "watchlist": remove_ticker(symbol)
    }
