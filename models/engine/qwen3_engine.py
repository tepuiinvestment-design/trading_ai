import json
import os

import numpy as np
import onnxruntime as ort

from models.tokenizer_qwen3 import Qwen3Tokenizer
from config.settings import settings

# Confirmed via session.get_inputs() on the 8B export: past_key_values.*.key/
# value are tensor(float) (fp32), same as the Phi-3 export, despite
# INT4-quantized weights. Both the 8B and 14B builds go through the same
# onnxruntime-genai export pipeline, so this is expected to hold for the 14B
# model too — re-confirm via session.get_inputs() during CPU validation.
KV_CACHE_DTYPE = np.float32

# Execution provider profiles. Flipping to GPU is just switching
# LUCY_EXECUTION_PROVIDER=gpu (or settings.MODEL_EXECUTION_PROVIDER) — no
# code change. NvTensorRtRtx falls back to CPU if unavailable.
PROVIDER_PROFILES = {
    "cpu": ["CPUExecutionProvider"],
    "gpu": ["NvTensorRtRtx", "CPUExecutionProvider"],
}


def resolve_providers(mode: str = None) -> list:
    mode = (mode or settings.MODEL_EXECUTION_PROVIDER or "cpu").lower()
    return PROVIDER_PROFILES.get(mode, PROVIDER_PROFILES["cpu"])


def _load_model_architecture(model_dir: str) -> dict:
    """Read layer/head/hidden-size architecture from the target model's own
    config file, instead of assuming one fixed shape.

    Previously this was hardcoded as module-level constants tuned for the 8B
    model (36 layers, 32 attention heads, 4096 hidden size) — silently wrong
    for any other model directory, e.g. the 14B build's 40 layers / 40 heads
    / 5120 hidden size (confirmed via models/qwen3_14b_cuda/genai_config.json
    and config.json, 2026-09-24).

    Prefers genai_config.json — the onnxruntime-genai config shipped
    alongside model.onnx, and the actual source of truth for this inference
    stack — and falls back to the raw HuggingFace config.json if that file
    isn't present in model_dir.
    """
    genai_config_path = os.path.join(model_dir, "genai_config.json")
    hf_config_path = os.path.join(model_dir, "config.json")

    if os.path.exists(genai_config_path):
        with open(genai_config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        model_cfg = raw["model"]
        decoder_cfg = model_cfg["decoder"]
        return {
            "num_hidden_layers": decoder_cfg["num_hidden_layers"],
            "num_attention_heads": decoder_cfg["num_attention_heads"],
            "num_key_value_heads": decoder_cfg["num_key_value_heads"],
            "head_dim": decoder_cfg["head_size"],
            "vocab_size": model_cfg["vocab_size"],
            "context_length": model_cfg["context_length"],
        }

    if os.path.exists(hf_config_path):
        with open(hf_config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        hidden_size = raw["hidden_size"]
        num_attention_heads = raw["num_attention_heads"]
        return {
            "num_hidden_layers": raw["num_hidden_layers"],
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": raw["num_key_value_heads"],
            # HF configs for Qwen3 include head_dim directly; fall back to
            # hidden_size / num_attention_heads for configs that don't.
            "head_dim": raw.get("head_dim", hidden_size // num_attention_heads),
            "vocab_size": raw["vocab_size"],
            "context_length": raw["max_position_embeddings"],
        }

    raise FileNotFoundError(
        f"No genai_config.json or config.json found in {model_dir!r} — "
        "cannot determine model architecture."
    )


class Qwen3Engine:
    def __init__(self, model_dir: str, onnx_model_name: str, execution_provider: str = None):
        self.model_path = os.path.join(model_dir, onnx_model_name)

        self.tokenizer = Qwen3Tokenizer()

        arch = _load_model_architecture(model_dir)

        # Clamp to the smaller of the two in case the tokenizer's real vocab
        # differs from the target model's own vocab_size.
        self.vocab_size = min(self.tokenizer.vocab_size, arch["vocab_size"])

        self.session = ort.InferenceSession(
            self.model_path,
            providers=resolve_providers(execution_provider),
        )

        self.input_ids_name = "input_ids"
        self.attention_mask_name = "attention_mask"

        self.num_layers = arch["num_hidden_layers"]
        self.num_attention_heads = arch["num_attention_heads"]  # not used for KV cache shape — GQA
        self.num_kv_heads = arch["num_key_value_heads"]
        self.head_dim = arch["head_dim"]
        self.context_length = arch["context_length"]

        self.kv_in_names = (
            [f"past_key_values.{i}.key" for i in range(self.num_layers)]
            + [f"past_key_values.{i}.value" for i in range(self.num_layers)]
        )
        self.kv_out_names = (
            [f"present.{i}.key" for i in range(self.num_layers)]
            + [f"present.{i}.value" for i in range(self.num_layers)]
        )

        self.logits_name = "logits"

    def _init_kv_cache(self):
        cache = {}
        for name in self.kv_in_names:
            cache[name] = np.zeros(
                (1, self.num_kv_heads, 0, self.head_dim),
                dtype=KV_CACHE_DTYPE,
            )
        return cache

    def _sample_next_token(
        self,
        logits_vector: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float = 1.0,
        previous_ids=None,
    ) -> int:
        # Ensure logits are limited to vocab size
        logits_vector = logits_vector[: self.vocab_size].astype(np.float64)

        # Penalize tokens already generated — guards against the degenerate
        # repetition loops this model falls into with long/complex prompts
        # at low temperature (e.g. repeating the same sentence verbatim).
        if repetition_penalty != 1.0 and previous_ids:
            for token_id in set(previous_ids):
                if token_id >= logits_vector.size:
                    continue
                if logits_vector[token_id] > 0:
                    logits_vector[token_id] /= repetition_penalty
                else:
                    logits_vector[token_id] *= repetition_penalty

        if temperature <= 0.0:
            return int(np.argmax(logits_vector))

        scaled = logits_vector / temperature
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        probs = probs / np.sum(probs)

        if top_k > 0 and top_k < probs.size:
            top_k_indices = np.argpartition(probs, -top_k)[-top_k:]
            mask = np.zeros_like(probs, dtype=bool)
            mask[top_k_indices] = True
            probs = np.where(mask, probs, 0.0)
            total = np.sum(probs)
            if total > 0:
                probs = probs / total

        if 0.0 < top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative = np.cumsum(sorted_probs)

            cutoff = cumulative <= top_p
            if not np.any(cutoff):
                cutoff[0] = True

            mask = np.zeros_like(probs, dtype=bool)
            mask[sorted_indices[cutoff]] = True
            probs = np.where(mask, probs, 0.0)
            total = np.sum(probs)
            if total > 0:
                probs = probs / total

        indices = np.arange(probs.size)
        return int(np.random.choice(indices, p=probs))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> str:
        # Reset KV cache per request
        kv_cache = self._init_kv_cache()
        past_seq_len = 0

        # Encode prompt and clip to context window
        ids = self.tokenizer.encode(prompt)
        ids = ids[: self.context_length]

        generated_ids = []

        for step in range(max_new_tokens):
            if step == 0:
                current_ids = ids
            else:
                current_ids = [generated_ids[-1]]

            arr = np.array([current_ids], dtype=np.int64)
            seq_len = arr.shape[1]

            total_seq_len = past_seq_len + seq_len
            if total_seq_len > self.context_length:
                break

            attention_mask = np.ones(
                (1, total_seq_len),
                dtype=np.int64,
            )

            feed = {
                self.input_ids_name: arr,
                self.attention_mask_name: attention_mask,
            }

            for name in self.kv_in_names:
                feed[name] = kv_cache[name]

            outputs = self.session.run(
                [self.logits_name] + self.kv_out_names,
                feed,
            )

            logits = outputs[0]
            logits_vector = logits[0, -1]

            # Limit logits to vocab size before sampling
            logits_vector = logits_vector[: self.vocab_size]

            next_id = self._sample_next_token(
                logits_vector,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                previous_ids=generated_ids,
            )

            # Extra safety: clamp to valid ID range
            if next_id < 0 or next_id >= self.vocab_size:
                next_id = int(np.argmax(logits_vector))

            for i, name in enumerate(self.kv_in_names):
                kv_cache[name] = outputs[i + 1]

            past_seq_len = total_seq_len
            generated_ids.append(next_id)

            if next_id in self.tokenizer.eos_ids:
                break

        # Decode only the newly generated continuation (not prompt+completion —
        # the old Phi-3 engine decoded the full sequence, which meant callers
        # got the prompt echoed back before the answer).
        return self.tokenizer.decode(generated_ids)
