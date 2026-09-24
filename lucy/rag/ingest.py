from lucy.rag.chunking import chunk_text
from lucy.rag.store import add_chunks, delete_by_source


def _ingest_text(ticker: str, source: str, text: str, doc_type: str, extra_metadata: dict = None):
    # Re-ingesting the same source overwrites its previous chunks.
    delete_by_source(source)

    chunks = chunk_text(text)
    if not chunks:
        return

    ids = [f"{source}::{i}" for i in range(len(chunks))]
    metadatas = []
    for i in range(len(chunks)):
        meta = {"ticker": ticker, "source": source, "doc_type": doc_type, "chunk_index": i}
        if extra_metadata:
            meta.update(extra_metadata)
        metadatas.append(meta)

    add_chunks(ids, chunks, metadatas)


def ingest_filing(ticker: str, source: str, text: str, extra_metadata: dict = None):
    _ingest_text(ticker, source, text, doc_type="filing", extra_metadata=extra_metadata)


def ingest_fundamentals(ticker: str, source: str, period: str, metrics: dict):
    lines = [f"Fundamentals for {ticker} ({period}):"]
    lines += [f"{key}: {value}" for key, value in metrics.items()]
    text = "\n".join(lines)
    _ingest_text(ticker, source, text, doc_type="fundamentals", extra_metadata={"period": period})


def ingest_market(ticker: str, source: str, snapshot: dict):
    lines = [f"Market data for {ticker}:"]
    lines += [f"{key}: {value}" for key, value in snapshot.items()]
    text = "\n".join(lines)
    _ingest_text(ticker, source, text, doc_type="market")


def ingest_performance(ticker: str, source: str, performance: dict):
    lines = [f"Historical price performance for {ticker}:"]
    lines += [f"{key}: {value}" for key, value in performance.items()]
    text = "\n".join(lines)
    _ingest_text(ticker, source, text, doc_type="performance")


def ingest_sector(ticker: str, source: str, sector_info: dict):
    lines = [f"Sector information for {ticker}:"]
    lines += [f"{key}: {value}" for key, value in sector_info.items()]
    text = "\n".join(lines)
    _ingest_text(ticker, source, text, doc_type="sector")


def ingest_news(ticker: str, source: str, articles: list):
    lines = [f"Recent news for {ticker}:"]
    has_unverified = False
    has_contradicted = False
    for article in articles:
        headline = article.get("headline")
        summary = article.get("summary")
        published = article.get("published")
        news_source = article.get("source")

        # `verification` is added by lucy.rag.news_verification.verify_articles
        # before ingestion (see lucy/analyze.py). Absent only if that step was
        # skipped, e.g. when re-ingesting older data — tag it explicitly
        # rather than silently presenting it as verified.
        verification = article.get("verification") or {}
        verdict = verification.get("verdict", "unchecked")
        confidence = verification.get("confidence")
        tag = verdict.upper() + (f" conf={confidence}" if confidence is not None else "")
        if verdict == "unverified":
            has_unverified = True
        elif verdict == "contradicted":
            has_contradicted = True

        lines.append(f"- [{published}] {headline} ({news_source}) [{tag}]: {summary}")
    text = "\n".join(lines)
    _ingest_text(
        ticker,
        source,
        text,
        doc_type="news",
        extra_metadata={
            "has_unverified_news": has_unverified,
            "has_contradicted_news": has_contradicted,
        },
    )
