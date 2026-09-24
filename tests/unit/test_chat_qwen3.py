from models.chat_qwen3 import build_chatml_prompt, parse_tool_calls


def test_build_chatml_prompt_basic_structure():
    messages = [
        {"role": "system", "content": "You are Lucy."},
        {"role": "user", "content": "What is AAPL doing?"},
    ]
    prompt = build_chatml_prompt(messages)

    assert "<|im_start|>system\nYou are Lucy.<|im_end|>\n" in prompt
    assert "<|im_start|>user\nWhat is AAPL doing?<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_enable_thinking_skips_suppression_block():
    messages = [{"role": "user", "content": "hi"}]
    prompt = build_chatml_prompt(messages, enable_thinking=True)
    assert "<think>\n\n</think>\n\n" not in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_tools_are_injected_into_system_message():
    messages = [
        {"role": "system", "content": "You are Lucy."},
        {"role": "user", "content": "What's the price?"},
    ]
    tools = [{"type": "function", "function": {"name": "get_price", "parameters": {}}}]
    prompt = build_chatml_prompt(messages, tools=tools)

    assert "# Tools" in prompt
    assert "get_price" in prompt
    assert "<tool_call>" in prompt


def test_tools_inserted_as_new_system_message_when_none_exists():
    messages = [{"role": "user", "content": "What's the price?"}]
    tools = [{"type": "function", "function": {"name": "get_price"}}]
    prompt = build_chatml_prompt(messages, tools=tools)

    assert prompt.startswith("<|im_start|>system\n")
    assert "get_price" in prompt


def test_parse_tool_calls_extracts_tagged_call():
    text = 'Some reasoning.\n<tool_call>\n{"name": "get_price", "arguments": {"symbol": "AAPL"}}\n</tool_call>\nDone.'
    cleaned, calls = parse_tool_calls(text)

    assert calls == [{"name": "get_price", "arguments": {"symbol": "AAPL"}}]
    assert "tool_call" not in cleaned
    assert "Some reasoning." in cleaned


def test_parse_tool_calls_no_calls_returns_full_text():
    text = "Just a plain answer with no tool calls."
    cleaned, calls = parse_tool_calls(text)
    assert calls == []
    assert cleaned == text


def test_parse_tool_calls_malformed_json_is_skipped():
    text = "<tool_call>\n{not valid json}\n</tool_call>"
    cleaned, calls = parse_tool_calls(text)
    assert calls == []


def test_parse_tool_calls_untagged_bare_json_fallback():
    """Real regression: this Qwen3 ONNX export sometimes emits a bare
    tool-call JSON object with no <tool_call> wrapper at all, even at
    temperature 0. Must still be recognized as a tool call, not leaked
    as raw JSON into the answer text."""
    text = '{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}'
    cleaned, calls = parse_tool_calls(text)

    assert calls == [{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}]
    assert cleaned == ""


def test_parse_tool_calls_bare_json_without_required_keys_not_treated_as_call():
    text = '{"foo": "bar"}'
    cleaned, calls = parse_tool_calls(text)
    assert calls == []
    assert cleaned == text
