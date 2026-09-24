"""Standalone CPU validation for the Qwen3-14B (INT4) ONNX build.

Mirrors the two-phase migration discipline used for the original Qwen3-8B
migration: validate the generalized engine end-to-end on CPU against the
already-downloaded models/qwen3_14b_cuda build, before the RTX 4000 is
physically installed and a GPU path is attempted.

This is a standalone smoke test, not a pytest — it loads real ~9.7GB model
weights and is slow on CPU (expect several minutes for even a short
completion on a 14B model), so it isn't meant to run in the regular test
suite. It does NOT touch models/phi3_generate.py's production 8B engine
instance.

Run from the project root:
    python validate_qwen3_14b_cpu.py
"""

import time

from models.engine.qwen3_engine import Qwen3Engine
from models.chat_qwen3 import build_chatml_prompt, parse_tool_calls

MODEL_DIR = r"C:\Users\tepui_66mr92q\trading_ai\models\qwen3_14b_cuda"


def main():
    print(f"Loading Qwen3-14B (INT4) from {MODEL_DIR} on CPU...")
    load_start = time.time()
    # Force CPU explicitly — this build is a CUDA/TensorRT-RTX export, but
    # NvTensorRtRtx has no fp8/T1000 relevance here; CPUExecutionProvider is
    # what actually validates the generalized engine logic before the RTX
    # 4000 arrives. See PROVIDER_PROFILES in qwen3_engine.py.
    engine = Qwen3Engine(
        model_dir=MODEL_DIR,
        onnx_model_name="model.onnx",
        execution_provider="cpu",
    )
    print(f"Loaded in {time.time() - load_start:.1f}s")
    print(f"Active providers: {engine.active_providers}  |  KV-cache dtype: {engine.kv_dtype.__name__}")

    # Sanity-check the architecture actually read from this model's own
    # config, not leftover 8B constants.
    print(
        f"num_layers={engine.num_layers} "
        f"num_attention_heads={engine.num_attention_heads} "
        f"num_kv_heads={engine.num_kv_heads} "
        f"head_dim={engine.head_dim} "
        f"context_length={engine.context_length} "
        f"vocab_size={engine.vocab_size}"
    )
    assert engine.num_layers == 40, f"expected 40 layers for 14B, got {engine.num_layers}"
    assert engine.num_attention_heads == 40, f"expected 40 attn heads for 14B, got {engine.num_attention_heads}"
    assert engine.head_dim == 128, f"expected head_dim 128, got {engine.head_dim}"
    import numpy as np
    assert engine.kv_dtype is np.float16, f"expected fp16 KV cache for the 14B CUDA build, got {engine.kv_dtype}"
    print("Architecture matches models/qwen3_14b_cuda/genai_config.json — engine is reading config, not hardcoded 8B values.")

    messages = [
        {"role": "system", "content": "You are Lucy, an autonomous AI market analyst. Be concise."},
        {"role": "user", "content": "In one sentence, what does a rising price-to-earnings ratio typically suggest about a stock?"},
    ]
    prompt = build_chatml_prompt(messages, enable_thinking=False)

    print("Generating (short completion, CPU — this will take a while on a 14B model)...")
    gen_start = time.time()
    completion = engine.generate(
        prompt=prompt,
        max_new_tokens=60,
        temperature=0.2,
        top_k=40,
        top_p=0.9,
    )
    gen_time = time.time() - gen_start

    text, tool_calls = parse_tool_calls(completion)
    print(f"\nGenerated in {gen_time:.1f}s ({gen_time / max(len(completion), 1):.2f}s/char, rough)")
    print(f"Tool calls parsed: {tool_calls}")
    print(f"\n--- Completion ---\n{text}\n------------------")

    if not text.strip():
        print("WARNING: empty completion — inspect KV cache shapes / tokenizer output before trusting this build.")
    else:
        print("Non-empty completion produced — engine + 14B model ran end-to-end on CPU.")


if __name__ == "__main__":
    main()
