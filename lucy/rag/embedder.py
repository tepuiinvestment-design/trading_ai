"""Embedding model for Lucy's RAG layer.

Default: BAAI/bge-small-en-v1.5, run through onnxruntime + tokenizers
(both already Lucy dependencies) — no sentence-transformers / PyTorch.
The model files live in models/bge_small_en_v15/ and are fetched once by
download_bge_model.py.

Legacy: ChromaDB's built-in MiniLM (all-MiniLM-L6-v2). This is what the RAG
layer actually used until 2026-09-24, despite earlier notes saying bge.

Pick with LUCY_EMBEDDER = "bge" (default) | "minilm". If "bge" is selected
but the model files haven't been downloaded yet, Lucy warns and falls back
to MiniLM so nothing breaks.

The two models produce vectors of the same size (384) but in different,
incompatible spaces, so each gets its own Chroma collection (see
store.collection_name()). Never mix them in one collection — Chroma
wouldn't error, retrieval would just silently return garbage.
"""

import os
import warnings
from pathlib import Path

import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions

try:  # chromadb >= 1.x: lets a persisted collection rebuild this EF by name
    from chromadb.utils.embedding_functions import register_embedding_function
except ImportError:  # pragma: no cover - older chromadb
    def register_embedding_function(cls):
        return cls

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
BGE_MODEL_DIR = PROJECT_ROOT / "models" / "bge_small_en_v15"
BGE_ONNX_RELPATH = Path("onnx") / "model.onnx"
BGE_MAX_TOKENS = 512

# bge v1.5's recommended instruction for short retrieval queries. Applied to
# queries only, never to stored passages (per the model card).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

EMBEDDER_BGE = "bge"
EMBEDDER_MINILM = "minilm"


def bge_model_files_present(model_dir: Path = BGE_MODEL_DIR) -> bool:
    model_dir = Path(model_dir)
    return (model_dir / BGE_ONNX_RELPATH).exists() and (model_dir / "tokenizer.json").exists()


@register_embedding_function
class BgeOnnxEmbeddingFunction(EmbeddingFunction[Documents]):
    """bge-small-en-v1.5 via onnxruntime: CLS pooling + L2 normalization,
    matching the model card's reference usage."""

    def __init__(self, model_dir: str = None, batch_size: int = 32, session=None, tokenizer=None):
        self.model_dir = Path(model_dir) if model_dir else BGE_MODEL_DIR
        self.batch_size = batch_size

        if tokenizer is None:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            tokenizer.enable_truncation(max_length=BGE_MAX_TOKENS)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer = tokenizer

        if session is None:
            import onnxruntime as ort

            # Deliberately CPU: the embedder is tiny, and keeping it off the
            # GPU leaves all VRAM for the LLM once the RTX 4000 is in.
            session = ort.InferenceSession(
                str(self.model_dir / BGE_ONNX_RELPATH),
                providers=["CPUExecutionProvider"],
            )
        self.session = session
        self._input_names = {i.name for i in self.session.get_inputs()}

    def _embed(self, texts: list) -> Embeddings:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer.encode_batch(batch)

            feed = {}
            if "input_ids" in self._input_names:
                feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
            if "attention_mask" in self._input_names:
                feed["attention_mask"] = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.array([e.type_ids for e in encoded], dtype=np.int64)

            last_hidden_state = self.session.run(None, feed)[0]  # (batch, seq, hidden)
            cls = last_hidden_state[:, 0, :].astype(np.float32)
            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            cls = cls / np.clip(norms, 1e-12, None)
            vectors.extend(cls.tolist())
        return vectors

    def __call__(self, input: Documents) -> Embeddings:
        return self._embed(list(input))

    def embed_query(self, input: Documents) -> Embeddings:
        return self._embed([BGE_QUERY_INSTRUCTION + text for text in input])

    @staticmethod
    def name() -> str:
        return "lucy_bge_small_en_v15_onnx"

    def get_config(self) -> dict:
        return {"model_dir": str(self.model_dir), "batch_size": self.batch_size}

    @staticmethod
    def build_from_config(config: dict) -> "BgeOnnxEmbeddingFunction":
        return BgeOnnxEmbeddingFunction(
            model_dir=config.get("model_dir"),
            batch_size=config.get("batch_size", 32),
        )


def active_embedder() -> str:
    """Which embedder this process will actually use, after the
    missing-model fallback."""
    choice = os.environ.get("LUCY_EMBEDDER", EMBEDDER_BGE).lower()
    if choice == EMBEDDER_MINILM:
        return EMBEDDER_MINILM
    if not bge_model_files_present():
        warnings.warn(
            f"{BGE_MODEL_ID} not found in {BGE_MODEL_DIR} — falling back to "
            "ChromaDB's MiniLM embedder. Run `python download_bge_model.py` "
            "to enable bge.",
            RuntimeWarning,
        )
        return EMBEDDER_MINILM
    return EMBEDDER_BGE


_embedding_function = None
_embedding_function_kind = None


def get_embedding_function():
    """Module-level singleton — the embedding model loads once per process,
    not once per request."""
    global _embedding_function, _embedding_function_kind
    if _embedding_function is None:
        _embedding_function_kind = active_embedder()
        if _embedding_function_kind == EMBEDDER_BGE:
            _embedding_function = BgeOnnxEmbeddingFunction()
        else:
            _embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return _embedding_function


def get_embedding_function_kind() -> str:
    get_embedding_function()
    return _embedding_function_kind
