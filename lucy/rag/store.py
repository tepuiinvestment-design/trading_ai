import os
from pathlib import Path

import chromadb

from lucy.rag.embedder import (
    EMBEDDER_BGE,
    get_embedding_function,
    get_embedding_function_kind,
)

# The original MiniLM-embedded collection. Kept (not deleted) so the
# migration script can copy its documents across and so LUCY_EMBEDDER=minilm
# still works as a rollback.
LEGACY_COLLECTION_NAME = "lucy_documents"
BGE_COLLECTION_NAME = "lucy_documents_bge_small_en_v15"

_client = None
_collection = None


def collection_name(embedder_kind: str) -> str:
    """One collection per embedding model — vectors from different models
    must never share a collection."""
    return BGE_COLLECTION_NAME if embedder_kind == EMBEDDER_BGE else LEGACY_COLLECTION_NAME


def _persist_dir() -> str:
    return os.environ.get("LUCY_RAG_PERSIST_DIR", "data/vectorstore")


def get_client():
    global _client
    if _client is None:
        persist_dir = _persist_dir()
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=persist_dir)
    return _client


def get_collection():
    """Module-level singleton — the Chroma client/collection is opened once
    per process, using whatever LUCY_RAG_PERSIST_DIR / LUCY_EMBEDDER are set
    to at first call."""
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=collection_name(get_embedding_function_kind()),
            embedding_function=get_embedding_function(),
        )
    return _collection


def delete_by_source(source: str):
    collection = get_collection()
    collection.delete(where={"source": source})


def add_chunks(ids: list, texts: list, metadatas: list):
    if not texts:
        return
    collection = get_collection()
    collection.add(ids=ids, documents=texts, metadatas=metadatas)


def query(query_text: str, ticker: str = None, k: int = 5, exclude_doc_types: list = None):
    collection = get_collection()

    clauses = []
    if ticker:
        clauses.append({"ticker": ticker})
    if exclude_doc_types:
        clauses.append({"doc_type": {"$nin": exclude_doc_types}})

    if not clauses:
        where = None
    elif len(clauses) == 1:
        where = clauses[0]
    else:
        where = {"$and": clauses}

    return collection.query(query_texts=[query_text], n_results=k, where=where)
