"""Copy everything in the old MiniLM-embedded collection ("lucy_documents")
into the new bge collection, re-embedding it with bge-small-en-v1.5.

No re-fetching from SEC EDGAR / Yahoo etc. — the stored chunk text and
metadata are reused as-is; only the vectors are recomputed. The old
collection is left untouched, so LUCY_EMBEDDER=minilm still works as a
rollback.

Safe to re-run: chunks already in the bge collection are overwritten
(upsert by the same id), not duplicated.

Run from the project root (after download_bge_model.py):
    python migrate_vectorstore_to_bge.py
"""

import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("LUCY_RAG_PERSIST_DIR", str(PROJECT_ROOT / "data" / "vectorstore"))

from lucy.rag.embedder import BgeOnnxEmbeddingFunction, bge_model_files_present  # noqa: E402
from lucy.rag.store import BGE_COLLECTION_NAME, LEGACY_COLLECTION_NAME, get_client  # noqa: E402

PAGE_SIZE = 256


def main():
    if not bge_model_files_present():
        raise SystemExit("bge model not downloaded yet — run: python download_bge_model.py")

    client = get_client()
    existing = {c.name if hasattr(c, "name") else c for c in client.list_collections()}
    if LEGACY_COLLECTION_NAME not in existing:
        print(f"No legacy collection '{LEGACY_COLLECTION_NAME}' found — nothing to migrate.")
        return

    legacy = client.get_collection(LEGACY_COLLECTION_NAME)
    target = client.get_or_create_collection(
        name=BGE_COLLECTION_NAME,
        embedding_function=BgeOnnxEmbeddingFunction(),
    )

    total = legacy.count()
    print(f"Migrating {total} chunks: '{LEGACY_COLLECTION_NAME}' (MiniLM) -> '{BGE_COLLECTION_NAME}' (bge-small-en-v1.5)")
    start = time.time()
    done = 0
    for offset in range(0, total, PAGE_SIZE):
        page = legacy.get(limit=PAGE_SIZE, offset=offset, include=["documents", "metadatas"])
        if not page["ids"]:
            break
        target.upsert(ids=page["ids"], documents=page["documents"], metadatas=page["metadatas"])
        done += len(page["ids"])
        print(f"  {done}/{total}")

    print(f"\nDone in {time.time() - start:.1f}s — '{BGE_COLLECTION_NAME}' now has {target.count()} chunks.")
    if target.count() != total:
        print("WARNING: counts differ — check the output above.")


if __name__ == "__main__":
    main()
