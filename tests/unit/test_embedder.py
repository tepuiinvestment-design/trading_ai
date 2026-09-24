"""Unit tests for lucy/rag/embedder.py (bge-small-en-v1.5 via onnxruntime)
and the per-embedder collection naming in lucy/rag/store.py.

Uses fake tokenizer/session objects, so no model download, no onnxruntime
session and no Chroma client are needed — these run fast anywhere.
"""

import numpy as np
import pytest

from lucy.rag import embedder as emb
from lucy.rag.embedder import (
    BGE_QUERY_INSTRUCTION,
    EMBEDDER_BGE,
    EMBEDDER_MINILM,
    BgeOnnxEmbeddingFunction,
)
from lucy.rag.store import BGE_COLLECTION_NAME, LEGACY_COLLECTION_NAME, collection_name


class _FakeEncoding:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)
        self.type_ids = [0] * len(ids)


class _FakeTokenizer:
    """Encodes each text as [len(text), 1, 2] so tests can tell inputs apart."""

    def __init__(self):
        self.seen = []

    def encode_batch(self, texts):
        self.seen.extend(texts)
        return [_FakeEncoding([len(t), 1, 2]) for t in texts]


class _FakeInput:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    """Returns last_hidden_state whose CLS vector is [input_ids[0], 3, 4, ...]
    and whose non-CLS tokens are large junk, so CLS pooling is observable."""

    def __init__(self, input_names=("input_ids", "attention_mask", "token_type_ids"), hidden=4):
        self._inputs = [_FakeInput(n) for n in input_names]
        self.hidden = hidden
        self.feeds = []

    def get_inputs(self):
        return self._inputs

    def run(self, output_names, feed):
        self.feeds.append(feed)
        ids = feed["input_ids"]
        batch, seq = ids.shape
        out = np.full((batch, seq, self.hidden), 1000.0, dtype=np.float32)
        for b in range(batch):
            out[b, 0, :] = [float(ids[b, 0]), 3.0, 4.0, 0.0][: self.hidden]
        return [out]


def _make(session=None, batch_size=32):
    return BgeOnnxEmbeddingFunction(
        tokenizer=_FakeTokenizer(),
        session=session or _FakeSession(),
        batch_size=batch_size,
    )


def test_uses_cls_token_and_l2_normalizes():
    ef = _make()
    [vec] = ef(["abcd"])  # CLS = [4, 3, 4, 0] -> norm sqrt(41)
    expected = np.array([4.0, 3.0, 4.0, 0.0]) / np.sqrt(41.0)
    np.testing.assert_allclose(np.asarray(vec), expected, rtol=1e-6)
    assert np.linalg.norm(vec) == pytest.approx(1.0)


def test_passages_are_not_prefixed_but_queries_are():
    ef = _make()
    ef(["passage text"])
    ef.embed_query(["what is AAPL's margin?"])
    assert ef.tokenizer.seen == [
        "passage text",
        BGE_QUERY_INSTRUCTION + "what is AAPL's margin?",
    ]


def test_batches_large_inputs():
    session = _FakeSession()
    ef = _make(session=session, batch_size=2)
    vectors = ef(["a", "bb", "ccc", "dddd", "eeeee"])
    assert len(vectors) == 5
    assert [f["input_ids"].shape[0] for f in session.feeds] == [2, 2, 1]


def test_only_feeds_inputs_the_graph_declares():
    """Some ONNX exports drop token_type_ids; feeding an undeclared input
    would make onnxruntime raise."""
    session = _FakeSession(input_names=("input_ids", "attention_mask"))
    ef = _make(session=session)
    ef(["x"])
    assert set(session.feeds[0]) == {"input_ids", "attention_mask"}


def test_config_roundtrip_names_the_model():
    ef = _make()
    assert BgeOnnxEmbeddingFunction.name() == "lucy_bge_small_en_v15_onnx"
    assert ef.get_config()["batch_size"] == 32


# --- embedder selection / fallback -------------------------------------------

def test_minilm_selected_explicitly(monkeypatch):
    monkeypatch.setenv("LUCY_EMBEDDER", "minilm")
    assert emb.active_embedder() == EMBEDDER_MINILM


def test_bge_is_default_when_model_present(monkeypatch):
    monkeypatch.delenv("LUCY_EMBEDDER", raising=False)
    monkeypatch.setattr(emb, "bge_model_files_present", lambda *a, **k: True)
    assert emb.active_embedder() == EMBEDDER_BGE


def test_falls_back_to_minilm_with_warning_when_bge_missing(monkeypatch):
    monkeypatch.delenv("LUCY_EMBEDDER", raising=False)
    monkeypatch.setattr(emb, "bge_model_files_present", lambda *a, **k: False)
    with pytest.warns(RuntimeWarning, match="download_bge_model"):
        assert emb.active_embedder() == EMBEDDER_MINILM


def test_model_files_present_checks_onnx_and_tokenizer(tmp_path):
    assert not emb.bge_model_files_present(tmp_path)
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"x")
    assert not emb.bge_model_files_present(tmp_path)  # tokenizer still missing
    (tmp_path / "tokenizer.json").write_text("{}")
    assert emb.bge_model_files_present(tmp_path)


# --- one collection per embedding model --------------------------------------

def test_each_embedder_gets_its_own_collection():
    """MiniLM and bge both emit 384-dim vectors, so Chroma would NOT error if
    they were mixed — retrieval would just silently degrade. Guard it."""
    assert collection_name(EMBEDDER_BGE) == BGE_COLLECTION_NAME
    assert collection_name(EMBEDDER_MINILM) == LEGACY_COLLECTION_NAME
    assert BGE_COLLECTION_NAME != LEGACY_COLLECTION_NAME
