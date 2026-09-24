"""Unit tests for models/engine/qwen3_engine.py's config-driven architecture
loading. Pure JSON-parsing logic — no model.onnx weights or onnxruntime
session involved, so these run fast and don't need any hardware.

Regression coverage for the bug this replaced: NUM_HIDDEN_LAYERS /
NUM_ATTENTION_HEADS / etc. used to be hardcoded module constants tuned for
the 8B model, which silently produced wrong KV-cache shapes for any other
model directory (e.g. the 14B build's 40 layers vs the 8B's 36).
"""

import json

import pytest

from models.engine.qwen3_engine import _load_model_architecture


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_reads_architecture_from_genai_config_14b_shape(tmp_path):
    """Real values from models/qwen3_14b_cuda/genai_config.json (2026-09-24)."""
    _write_json(
        tmp_path / "genai_config.json",
        {
            "model": {
                "vocab_size": 151936,
                "context_length": 40960,
                "decoder": {
                    "num_hidden_layers": 40,
                    "num_attention_heads": 40,
                    "num_key_value_heads": 8,
                    "head_size": 128,
                },
            }
        },
    )

    arch = _load_model_architecture(str(tmp_path))

    assert arch == {
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "context_length": 40960,
    }


def test_reads_architecture_from_genai_config_8b_shape(tmp_path):
    """Regression check: the previously-hardcoded 8B constants (36 layers,
    32 attention heads, 4096 hidden size -> head_dim 128) must still come
    out correctly now that they're read from config instead."""
    _write_json(
        tmp_path / "genai_config.json",
        {
            "model": {
                "vocab_size": 151936,
                "context_length": 40960,
                "decoder": {
                    "num_hidden_layers": 36,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "head_size": 128,
                },
            }
        },
    )

    arch = _load_model_architecture(str(tmp_path))

    assert arch["num_hidden_layers"] == 36
    assert arch["num_attention_heads"] == 32
    assert arch["head_dim"] == 128


def test_falls_back_to_hf_config_when_genai_config_missing(tmp_path):
    """genai_config.json is preferred, but a directory with only the raw
    HuggingFace config.json (e.g. a freshly-downloaded, not-yet-genai-
    converted model) should still work."""
    _write_json(
        tmp_path / "config.json",
        {
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_size": 5120,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
        },
    )

    arch = _load_model_architecture(str(tmp_path))

    assert arch == {
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "context_length": 40960,
    }


def test_hf_config_derives_head_dim_when_absent(tmp_path):
    """Some HF configs omit head_dim explicitly; derive it from
    hidden_size / num_attention_heads instead of failing."""
    _write_json(
        tmp_path / "config.json",
        {
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
        },
    )

    arch = _load_model_architecture(str(tmp_path))

    assert arch["head_dim"] == 128  # 4096 / 32


def test_genai_config_preferred_over_hf_config_when_both_present(tmp_path):
    _write_json(
        tmp_path / "genai_config.json",
        {
            "model": {
                "vocab_size": 111,
                "context_length": 222,
                "decoder": {
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 3,
                    "head_size": 4,
                },
            }
        },
    )
    _write_json(
        tmp_path / "config.json",
        {
            "num_hidden_layers": 999,
            "num_attention_heads": 999,
            "num_key_value_heads": 999,
            "head_dim": 999,
            "vocab_size": 999,
            "max_position_embeddings": 999,
        },
    )

    arch = _load_model_architecture(str(tmp_path))

    assert arch["num_hidden_layers"] == 1  # from genai_config.json, not config.json


def test_raises_when_no_config_file_present(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_model_architecture(str(tmp_path))


# --- KV-cache dtype read from the ONNX graph (2026-09-24) -------------------
# The 8B CPU build takes fp32 KV cache; the 14B CUDA build takes fp16. The
# old hardcoded fp32 constant would crash the 14B build on its first run.

import numpy as np

from models.engine.qwen3_engine import (
    TRT_RTX_PROVIDER,
    _kv_cache_dtype_from_session,
    resolve_providers,
)


class _FakeInput:
    def __init__(self, name, type_):
        self.name = name
        self.type = type_


class _FakeSession:
    def __init__(self, kv_type, num_layers):
        self._inputs = [
            _FakeInput("input_ids", "tensor(int64)"),
            _FakeInput("attention_mask", "tensor(int64)"),
        ]
        for i in range(num_layers):
            self._inputs.append(_FakeInput(f"past_key_values.{i}.key", kv_type))
            self._inputs.append(_FakeInput(f"past_key_values.{i}.value", kv_type))

    def get_inputs(self):
        return self._inputs


def _kv_names(n):
    return [f"past_key_values.{i}.key" for i in range(n)] + [
        f"past_key_values.{i}.value" for i in range(n)
    ]


def test_kv_dtype_fp32_for_8b_cpu_build():
    session = _FakeSession("tensor(float)", 36)
    assert _kv_cache_dtype_from_session(session, _kv_names(36)) is np.float32


def test_kv_dtype_fp16_for_14b_cuda_build():
    session = _FakeSession("tensor(float16)", 40)
    assert _kv_cache_dtype_from_session(session, _kv_names(40)) is np.float16


def test_kv_dtype_raises_when_config_layers_exceed_graph():
    """Config says 40 layers but graph only has 36 -> fail loudly, not
    silently mis-shape the cache."""
    session = _FakeSession("tensor(float)", 36)
    with pytest.raises(ValueError, match="missing"):
        _kv_cache_dtype_from_session(session, _kv_names(40))


# --- Execution provider resolution ------------------------------------------

def test_cpu_mode_resolves_to_cpu():
    assert resolve_providers("cpu", available=["CPUExecutionProvider"]) == [
        "CPUExecutionProvider"
    ]


def test_gpu_mode_uses_trt_rtx_when_available():
    available = [TRT_RTX_PROVIDER, "CPUExecutionProvider"]
    assert resolve_providers("gpu", available=available) == [
        TRT_RTX_PROVIDER,
        "CPUExecutionProvider",
    ]


def test_gpu_mode_warns_and_falls_back_when_trt_rtx_missing():
    """Plain CPU onnxruntime wheel: must warn instead of silently running
    on CPU while claiming GPU."""
    with pytest.warns(RuntimeWarning, match="not available"):
        providers = resolve_providers("gpu", available=["CPUExecutionProvider"])
    assert providers == ["CPUExecutionProvider"]
