"""GPU validation for the Qwen3-14B (INT4) ONNX build, on the current T1000
(4GB VRAM) — run at the user's request instead of the CPU path.

Heads up: models/qwen3_14b_cuda/model.onnx.data alone is ~9.7GB on disk.
INT4 quantization means VRAM usage is well below that, but ~9.7GB of INT4 +
some INT8-override weights (see the model's config.json quantization_config
overrides) is still a lot to fit in 4GB total VRAM alongside the KV cache
and TensorRT-RTX's own working memory. This may simply fail with an
out-of-memory error on the T1000 — that's expected to be possible here and
is itself useful information: it tells us whether we need to wait for the
RTX 4000 (20GB VRAM) before the 14B build is usable at all, or whether it
just barely fits today. See validate_qwen3_14b_cpu.py for the CPU-only
fallback that isolates engine/logic bugs from VRAM constraints.

This is a standalone smoke test, not a pytest — it loads real model weights
and is slow on first run (NvTensorRtRtx builds/caches a TensorRT engine the
first time a given model+shape combination runs). It does NOT touch
models/phi3_generate.py's production 8B engine instance, which still runs
CPU today.

Run from the project root:
    python validate_qwen3_14b_gpu.py
"""

import sys
import time

from models.engine.qwen3_engine import Qwen3Engine
from models.chat_qwen3 import build_chatml_prompt, parse_tool_calls

MODEL_DIR = r"C:\Users\tepui_66mr92q\trading_ai\models\qwen3_14b_cuda"


def main():
    print(f"Loading Qwen3-14B (INT4) from {MODEL_DIR} on GPU (NvTensorRtRtx, falls back to CPU if unavailable)...")
    print("First load can be slow — TensorRT-RTX compiles/caches an engine for this model+shape the first time.")
    load_start = time.time()
    try:
        engine = Qwen3Engine(
            model_dir=MODEL_DIR,
            onnx_model_name="model.onnx",
            execution_provider="gpu",
        )
    except Exception as exc:
        print(f"\nFAILED to load on GPU after {time.time() - load_start:.1f}s: {exc}")
        print(
            "If this looks like an out-of-memory error, the T1000's 4GB VRAM likely "
            "can't fit the 14B build — that's a real answer (wait for the RTX 4000), "
            "not a bug. Try validate_qwen3_14b_cpu.py to confirm the engine/logic "
            "itself is fine independent of VRAM."
        )
        sys.exit(1)
    print(f"Loaded in {time.time() - load_start:.1f}s")

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
    print("Architecture matches models/qwen3_14b_cuda/genai_config.json — engine is reading config, not hardcoded 8B values.")

    messages = [
        {"role": "system", "content": "You are Lucy, an autonomous AI market analyst. Be concise."},
        {"role": "user", "content": "In one sentence, what does a rising price-to-earnings ratio typically suggest about a stock?"},
    ]
    prompt = build_chatml_prompt(messages, enable_thinking=False)

    print("Generating (short completion)...")
    gen_start = time.time()
    try:
        completion = engine.generate(
            prompt=prompt,
            max_new_tokens=60,
            temperature=0.2,
            top_k=40,
            top_p=0.9,
        )
    except Exception as exc:
        print(f"\nFAILED during generation after {time.time() - gen_start:.1f}s: {exc}")
        print(
            "Load succeeded but generation failed — this can happen if VRAM runs out "
            "once the KV cache grows (load-time success doesn't guarantee headroom "
            "for a full generation). Also worth checking against validate_qwen3_14b_cpu.py."
        )
        sys.exit(1)
    gen_time = time.time() - gen_start

    text, tool_calls = parse_tool_calls(completion)
    print(f"\nGenerated in {gen_time:.1f}s")
    print(f"Tool calls parsed: {tool_calls}")
    print(f"\n--- Completion ---\n{text}\n------------------")

    if not text.strip():
        print("WARNING: empty completion — inspect KV cache shapes / tokenizer output before trusting this build.")
    else:
        print("Non-empty completion produced — engine + 14B model ran end-to-end on GPU (T1000).")


if __name__ == "__main__":
    main()
