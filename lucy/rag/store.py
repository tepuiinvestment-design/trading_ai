import os
from pathlib import Path

import chromadb

from lucy.rag.embedder import get_embedding_function

COLLECTION_NAME = "lucy_documents"

_client = None
_collection = None


def _persist_dir() -> str:
    return os.environ.get("LUCY_RAG_PERSIST_DIR", "data/vectorstore")


def get_collection():
    """Module-level singleton — the Chroma client/collection is opened once
    per process, using whatever LUCY_RAG_PERSIST_DIR is set to at first call."""
    global _client, _collection
    if _collection is None:
        persist_dir = _persist_dir()
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=persist_dir)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
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
