import os
from pathlib import Path

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("LUCY_RAG_PERSIST_DIR", str(PROJECT_ROOT / "data" / "vectorstore"))
os.environ.setdefault("LUCY_WATCHLIST_PATH", str(PROJECT_ROOT / "data" / "watchlist.csv"))

from api.router import router

app = FastAPI(
    title="Lucy AI Backend",
    description="FastAPI backend for Lucy, the autonomous AI market analyst.",
    version="1.0.0"
)

# Attach all routers (Lucy endpoints will live inside api/router.py)
app.include_router(router)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Lucy backend is running."
    }
