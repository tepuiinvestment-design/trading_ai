from models.engine.qwen3_engine import Qwen3Engine
from models.chat_qwen3 import build_chatml_prompt, parse_tool_calls
from config.settings import settings

# Model directory is config-driven via settings.MODEL_DIR (LUCY_MODEL_DIR env
# var) — flip to models\qwen3_14b_cuda once it's validated, same way
# LUCY_EXECUTION_PROVIDER flips CPU/GPU, no code change needed either way.
engine = Qwen3Engine(
    model_dir=settings.MODEL_DIR,
    onnx_model_name="model.onnx",
)


def generate_api(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
) -> str:
    return engine.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


def generate_chat(
    messages: list,
    tools: list = None,
    enable_thinking: bool = False,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
) -> dict:
    """Format `messages` as Qwen3 ChatML, generate, and split out any
    <tool_call> blocks. Returns {"text": str, "tool_calls": list[dict]}."""
    prompt = build_chatml_prompt(messages, tools=tools, enable_thinking=enable_thinking)
    completion = engine.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    text, tool_calls = parse_tool_calls(completion)
    return {"text": text, "tool_calls": tool_calls}
