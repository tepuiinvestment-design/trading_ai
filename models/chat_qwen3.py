import json
import re

# Qwen3's documented tool-calling convention (Hermes-style function calling):
# tool schemas are dumped into the system message inside <tools>...</tools>,
# and the model is instructed to reply with <tool_call>{...}</tool_call>.
TOOL_SYSTEM_BLOCK = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "<tools>\n{tool_lines}\n</tools>\n\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{{"name": <function-name>, "arguments": <args-json-object>}}\n'
    "</tool_call>"
)

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def build_chatml_prompt(messages, tools=None, enable_thinking: bool = False) -> str:
    """Render a list of {"role", "content"} messages as a Qwen3 ChatML prompt.

    Qwen3 emits a <think>...</think> reasoning block before its answer by
    default — on CPU this alone can burn the entire max_new_tokens budget
    before any answer text appears. enable_thinking=False (the default here)
    forces the empty-think-block suppression the model's own chat template
    defines, so short generations still produce an answer.
    """
    messages = list(messages)

    if tools:
        tool_lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
        tools_block = TOOL_SYSTEM_BLOCK.format(tool_lines=tool_lines)

        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"] + "\n\n" + tools_block,
            }
        else:
            messages.insert(0, {"role": "system", "content": tools_block})

    parts = [
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    ]
    parts.append("<|im_start|>assistant\n")
    if not enable_thinking:
        parts.append("<think>\n\n</think>\n\n")
    return "".join(parts)


def parse_tool_calls(text: str):
    """Split model output into (cleaned_text, tool_calls). tool_calls is a
    list of {"name": ..., "arguments": ...} dicts parsed out of any
    <tool_call>...</tool_call> blocks; malformed blocks are skipped.

    Fallback: this Qwen3 ONNX export sometimes emits a bare tool-call JSON
    object with no <tool_call> wrapper at all (observed even at temperature
    0), despite the system prompt instructing it to use the tags. If no
    tagged calls are found and the entire response is exactly one JSON
    object shaped like {"name": ..., "arguments": ...}, treat it as an
    untagged tool call rather than leaking raw JSON into the answer text.
    """
    calls = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue

    cleaned = TOOL_CALL_PATTERN.sub("", text).strip()

    if not calls and cleaned:
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            calls.append(obj)
            cleaned = ""

    return cleaned, calls
