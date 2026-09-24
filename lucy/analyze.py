from web.fetch_market import fetch_market_price
from web.fetch_fundamentals import fetch_fundamentals
from web.fetch_sector import fetch_sector
from web.fetch_news import fetch_news
from web.fetch_performance import fetch_performance
from models.phi3_generate import generate_chat
from lucy.rag import retrieve_context, ingest, verify_articles
from lucy.watchlist import is_watched
from lucy.grounding import check_grounding
from lucy.metric_labels import label_fundamentals

def run_full_analysis(symbol: str):
    symbol = symbol.upper()

    if not is_watched(symbol):
        return {
            "status": "error",
            "symbol": symbol,
            "message": f"{symbol} is not on the watchlist.",
        }

    market = fetch_market_price(symbol)
    fundamentals = fetch_fundamentals(symbol)
    sector = fetch_sector(symbol)
    news = fetch_news(symbol)
    performance = fetch_performance(symbol)

    # Scan news for unverifiable/spoofed sources and any share-price claim
    # that contradicts the market data just fetched, before that news ever
    # reaches the RAG store or the model. See lucy/rag/news_verification.py
    # for the source-credibility and contradiction checks involved.
    if news.get("status") == "ok" and news.get("news"):
        current_price = market.get("price") if market.get("status") == "ok" else None
        news["news"] = verify_articles(news["news"], current_price=current_price)

    # Pre-compute high/low/moderate labels in Python rather than asking the
    # model to apply the thresholds itself — it was unreliable at that even
    # when given the exact reference ranges (e.g. calling a 79.5, "moderate"
    # per our own scale, debt-to-equity "high"). Relaying a given label is
    # a much more reliable task for it than deriving one.
    if fundamentals.get("status") == "ok":
        fundamentals = {**fundamentals, **label_fundamentals(fundamentals)}

    # Feed every freshly fetched, real data source into the RAG store so
    # retrieval has actual facts to draw on — not just fundamentals. Safe to
    # re-run: ingestion overwrites the prior chunks for the same source.
    if market.get("status") == "ok":
        snapshot = {k: v for k, v in market.items() if k not in ("status", "symbol", "source")}
        ingest.ingest_market(ticker=symbol, source=f"{symbol}_market_latest", snapshot=snapshot)

    if fundamentals.get("status") == "ok":
        metrics = {k: v for k, v in fundamentals.items() if k not in ("status", "symbol", "source")}
        ingest.ingest_fundamentals(
            ticker=symbol,
            source=f"{symbol}_fundamentals_latest",
            period="latest",
            metrics=metrics,
        )

    if sector.get("status") == "ok":
        sector_info = {k: v for k, v in sector.items() if k not in ("status", "symbol")}
        ingest.ingest_sector(ticker=symbol, source=f"{symbol}_sector_latest", sector_info=sector_info)

    if news.get("status") == "ok" and news.get("news"):
        ingest.ingest_news(ticker=symbol, source=f"{symbol}_news_latest", articles=news["news"])

    if performance.get("status") == "ok":
        perf_metrics = {k: v for k, v in performance.items() if k not in ("status", "symbol", "source")}
        ingest.ingest_performance(ticker=symbol, source=f"{symbol}_performance_latest", performance=perf_metrics)

    # Filings are excluded here — they're real but dense (accounting-standard
    # boilerplate etc.), and blending them into every routine analysis bloats
    # the prompt enough to trigger degenerate repetition loops. They stay in
    # the RAG store for a future, more targeted retrieval use case.
    context = retrieve_context(
        f"Market data, fundamentals, sector, and recent news for {symbol}",
        ticker=symbol,
        k=6,
        exclude_doc_types=["filing"],
    )

    system_content = (
        "You are Lucy, an autonomous AI market analyst. Base your analysis "
        "strictly on the data and retrieved context provided below — do not "
        "invent numbers, statistics, dates, or facts that are not present in "
        "them. If a data point is missing or unavailable, say so explicitly "
        "rather than guessing or fabricating a value. Do not name any "
        "specific competing company, product, or service (e.g. a competitor's "
        "cloud or hardware product) unless it is explicitly named in the "
        "provided data or news below — refer to competitors generically "
        "(e.g. 'other major cloud providers') if none are named in the data. "
        "When discussing recent news, only reference the specific headlines "
        "provided below, not general background knowledge. Each news item "
        "carries a 'verification' field with a verdict and confidence score, "
        "computed independently of you: 'verified' means the publisher is "
        "one Lucy recognizes and nothing in it contradicts the fetched "
        "market data; 'unverified' means Lucy could not confirm who the "
        "publisher actually is; 'contradicted' means it cites a share price "
        "at odds with the real one fetched this run. Only present a "
        "'verified' item as settled fact. For 'unverified' items, either "
        "omit them or explicitly attribute them as an unconfirmed report "
        "from an unverified source — never state their content as fact. "
        "Never repeat a figure from a 'contradicted' item, and note that it "
        "conflicts with verified data if you mention it at all. The Market Data "
        "section below contains EXACTLY these four fields and nothing else: "
        "price, open, high, low — today's values only; do not describe "
        "today's price relative to a 52-week range from this section. The "
        "real 52-week high/low DOES live in the Fundamentals section as "
        "fifty_two_week_low/fifty_two_week_high — you may compare today's "
        "price to those specific fields, but never to a moving average or "
        "all-time high, since neither is provided anywhere. Several "
        "fundamentals metrics come with a pre-computed '_label' field right "
        "next to them (e.g. de_label, pe_label, analyst_rating_mean_label) "
        "— these are already correctly classified as high/low/moderate/etc. "
        "ALWAYS use that exact label when describing the metric. Never "
        "compute or guess your own high/low/strong/weak judgment for a "
        "metric that has a '_label' field — just relay the given label. "
        "Analyst rating/price targets are real professional consensus "
        "data, not your own opinion — present them as 'analysts rate this "
        "X' rather than as your own judgment. "
        "The Performance section gives real historical returns (1-month, "
        "YTD, 1-year, 3-year, 5-year). If any of these is null, that means "
        "the stock doesn't have enough trading history for that period "
        "(e.g. a recent IPO) — say so explicitly, do not estimate or guess "
        "a number for it. Be concise: use short bullet points (max 2 "
        "sentences each), not long paragraphs, so the full analysis — all "
        "five sections — fits within the response length."
    )
    if context:
        system_content += f"\n\nRelevant retrieved context:\n{context}"

    messages = [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": f"""Analyze the following data for {symbol}:

Market Data (today only — no 52-week range or moving average here; 52-week range is in Fundamentals below):
{market}

Fundamentals:
{fundamentals}

Historical Performance (real returns; null means not enough trading history for that period):
{performance}

Sector Information:
{sector}

Recent News:
{news}

Provide, as short bullet points (max 3 bullets per section, max 2 sentences per bullet):
- Key risks
- Key opportunities
- Historical performance (state the 1-month, YTD, 1-year, 3-year, and 5-year returns exactly as given; say "not available" for any null)
- Sector context
- Fundamental strengths/weaknesses
- A concise final verdict (1-2 sentences total)""",
        },
    ]

    result = generate_chat(
        messages,
        max_new_tokens=650,
        temperature=0.2,
        top_k=40,
        top_p=0.9
    )

    source_data = {"market": market, "fundamentals": fundamentals, "sector": sector, "performance": performance, "news": news}
    source_summary = f"""Market Data:
{market}

Fundamentals:
{fundamentals}

Historical Performance:
{performance}

Sector Information:
{sector}

Recent News:
{news}"""

    grounding_review = check_grounding(result["text"], source_data, source_summary)

    return {
        "status": "ok",
        "symbol": symbol,
        "market": market,
        "fundamentals": fundamentals,
        "performance": performance,
        "sector": sector,
        "news": news,
        "analysis": result["text"],
        "grounding_review": grounding_review,
        "tool_calls": result["tool_calls"]
    }
