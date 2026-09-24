"""Quick side-by-side retrieval check: the same query against the old MiniLM
collection and the new bge collection. Read-only — changes nothing.

Run from the project root:
    python check_rag_retrieval.py
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("LUCY_RAG_PERSIST_DIR", str(PROJECT_ROOT / "data" / "vectorstore"))

from chromadb.utils import embedding_functions  # noqa: E402

from lucy.rag.embedder import BgeOnnxEmbeddingFunction  # noqa: E402
from lucy.rag.store import BGE_COLLECTION_NAME, LEGACY_COLLECTION_NAME, get_client  # noqa: E402

QUERIES = [
    "What are the main risk factors in the latest annual report?",
    "How has revenue and operating margin changed recently?",
]


def show(label, collection, query):
    res = collection.query(query_texts=[query], n_results=3, include=["documents", "metadatas", "distances"])
    print(f"  [{label}]")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        snippet = " ".join(doc.split())[:110]
        print(f"    {dist:.3f}  {meta.get('ticker')}/{meta.get('doc_type')}: {snippet}")


def main():
    client = get_client()
    minilm = client.get_collection(LEGACY_COLLECTION_NAME, embedding_function=embedding_functions.DefaultEmbeddingFunction())
    bge = client.get_collection(BGE_COLLECTION_NAME, embedding_function=BgeOnnxEmbeddingFunction())
    for q in QUERIES:
        print(f"\nQUERY: {q}")
        show("MiniLM (old)", minilm, q)
        show("bge (new)", bge, q)


if __name__ == "__main__":
    main()
