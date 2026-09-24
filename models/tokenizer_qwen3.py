from pathlib import Path
from tokenizers import Tokenizer

TOKENIZER_JSON = Path(
    r"C:\Users\tepui_66mr92q\trading_ai\models\qwen3_8b\tokenizer.json"
)

# Qwen3 stop tokens (ChatML). Resolved to IDs at load time from the real
# vocab rather than hardcoded, since we can't confirm numeric IDs without
# the actual downloaded tokenizer.json.
EOS_TOKEN_STRINGS = ("<|im_end|>", "<|endoftext|>")


class Qwen3Tokenizer:
    def __init__(self, tokenizer_json_path: Path = TOKENIZER_JSON):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
        self.vocab_size = self.tokenizer.get_vocab_size()

        self.eos_ids = set()
        for tok in EOS_TOKEN_STRINGS:
            tok_id = self.tokenizer.token_to_id(tok)
            if tok_id is not None:
                self.eos_ids.add(tok_id)

        if not self.eos_ids:
            raise ValueError(
                f"None of {EOS_TOKEN_STRINGS} were found in the Qwen3 tokenizer "
                "vocab — check the tokenizer.json is the correct Qwen3 export."
            )

        # Kept for parity with the old Phi-3 tokenizer's single-id interface;
        # engine code should prefer eos_ids (a model can have >1 stop token).
        self.eos_id = next(iter(self.eos_ids))

    def encode(self, text: str):
        return self.tokenizer.encode(text).ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)
