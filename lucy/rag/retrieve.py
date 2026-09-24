from lucy.rag.store import query


def retrieve_context(user_query: str, ticker: str = None, k: int = 5, exclude_doc_types: list = None) -> str:
    """Return the top-k retrieved chunks joined as plain text, or "" if
    nothing relevant is in the store yet (e.g. an un-ingested ticker)."""
    try:
        results = query(user_query, ticker=ticker, k=k, exclude_doc_types=exclude_doc_types)
    except Exception:
        return ""

    documents = results.get("documents") or []
    if not documents or not documents[0]:
        return ""

    return "\n\n".join(documents[0])
