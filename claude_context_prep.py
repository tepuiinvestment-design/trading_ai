"""Local pre-digest tool: watches a folder for raw text/log/csv/json drops,
compresses each one with Lucy's own local model into a compact structured
JSON summary, and writes a Claude-ready briefing file — so a large raw dump
doesn't have to be pasted whole into a Claude Code session, burning cloud
context on data a local pass can compress first.

Rewired from an initial draft that assumed a separate Ollama server serving
qwen2.5:14b-instruct-fp8 over an OpenAI-compatible endpoint. This version
uses the SAME inference stack as the rest of Lucy instead: the shared
Qwen3Engine instance in models.phi3_generate, config-driven via
LUCY_MODEL_DIR / LUCY_EXECUTION_PROVIDER (see config/settings.py). No
second server, no new model download, no VRAM contention with anything
else — this is literally the same engine lucy/analyze.py uses.

Run from the project root:
    python claude_context_prep.py

Requires `watchdog` — see requirements_claude_prep_addition.txt.
"""

import os
import re
import time
import json

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from models.phi3_generate import generate_chat

WATCH_DIR = "./raw_data_input"
OUTPUT_DIR = "./claude_payload_output"

os.makedirs(WATCH_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The local model reads the whole file into one ChatML prompt rather than
# chunking it, so keep raw input modest. This is a char-count heuristic, not
# a token count — same approach lucy/analyze.py uses elsewhere.
MAX_CHARS = 24000

SYSTEM_PROMPT = (
    "You are a local data synthesis node. Compress unstructured raw text "
    "input into a dense, structured summary for a downstream analyst. "
    "Respond with ONLY a JSON object, no markdown fences, no commentary "
    "before or after it, matching exactly this schema:\n"
    "{\n"
    '  "statistical_anomalies": ["strings describing statistical irregularities or outliers"],\n'
    '  "dense_thematic_summary": "a concentrated compression of the raw text\'s context",\n'
    '  "underlying_structural_trends": ["strings describing key trend indicators"]\n'
    "}"
)


def _extract_json_object(text: str):
    """Best-effort JSON extraction. The raw ONNX Runtime path has no
    response_format=json_object enforcement — that's an OpenAI-server-only
    feature, not something onnxruntime-genai provides. This model is known
    to occasionally pad or wrap its JSON despite instructions not to (the
    same behavior models/chat_qwen3.py's parse_tool_calls already works
    around for tool calls), so try a direct parse first, then fall back to
    slicing out the outermost {...} block before giving up."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


class DataIngestionHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        file_name = os.path.basename(file_path)

        # Skip temp/hidden files and anything not one of our expected types.
        if file_name.startswith(".") or not file_path.endswith((".txt", ".log", ".csv", ".json")):
            return

        print(f"\n[!] Ingestion detected: {file_name}")
        time.sleep(1)  # let any in-progress write finish

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()

            if len(raw_content) > MAX_CHARS:
                print(f"[!] Payload exceeds {MAX_CHARS} chars — truncating.")
                raw_content = raw_content[:MAX_CHARS]

            print("[-] Compressing with Lucy's local model...")
            structured = self.compress_locally(raw_content)

            if structured is not None:
                self.write_claude_bundle(file_name, structured)
                print("[+] Claude briefing ready.")
            else:
                print("[x] Local model did not return parseable JSON — see raw output above.")

        except Exception as exc:
            print(f"[x] Pipeline error on {file_name}: {exc}")

    def compress_locally(self, raw_text: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze and compress the following payload:\n\n{raw_text}"},
        ]
        # Reuses the shared engine instance from models.phi3_generate — the
        # same model/provider Lucy's own analysis pipeline uses.
        result = generate_chat(
            messages,
            max_new_tokens=400,
            temperature=0.2,
            top_k=40,
            top_p=0.9,
        )
        text = result["text"]
        parsed = _extract_json_object(text)
        if parsed is None:
            print(f"--- raw model output ---\n{text}\n------------------------")
        return parsed

    def write_claude_bundle(self, original_filename: str, structured: dict):
        output_filename = f"READY_FOR_CLAUDE_{os.path.splitext(original_filename)[0]}.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        bundle = f"""[ROLE] You are the research director for this session.
[LOCAL DATA PROVIDER] The block below is a structured summary Lucy's local model extracted from a raw data drop.

<local_extracted_payload>
{json.dumps(structured, indent=2)}
</local_extracted_payload>

[TASK]
1. Identify the significant patterns across this data.
2. Note the operational implications.
3. Propose up to 3 hypotheses for the root cause of the trends above.
4. If useful, suggest what additional data Lucy should fetch next to confirm or rule out a hypothesis.
"""
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(bundle)

        print(f"[+] Wrote {output_path}")


if __name__ == "__main__":
    print("=" * 65)
    print(" Local Claude-context prep watcher")
    print("=" * 65)
    print(f"[*] Watching: {WATCH_DIR}")
    print(f"[*] Output:   {OUTPUT_DIR}")
    print("[*] Drop a .txt/.log/.csv/.json file into the input folder to process it.")

    event_handler = DataIngestionHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[-] Shutting down...")
        observer.stop()
    observer.join()
