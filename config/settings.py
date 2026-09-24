import os


class Settings:
    PROJECT_NAME: str = "Trading AI Backend"
    VERSION: str = "0.1.0"

    # Model execution provider profile: "cpu" (default, current hardware)
    # or "gpu" (NvTensorRtRtx with CPU fallback, once the RTX 4000 SFF Ada
    # is installed). Flip via env var — no code change needed.
    MODEL_EXECUTION_PROVIDER: str = os.environ.get("LUCY_EXECUTION_PROVIDER", "cpu")

    # Which model directory Qwen3Engine loads. Defaults to the CPU-validated
    # Qwen3-8B build. Set LUCY_MODEL_DIR to point at another model directory
    # (e.g. models\qwen3_14b_cuda, once it's been CPU/GPU-validated) without
    # touching code — same config-driven pattern as MODEL_EXECUTION_PROVIDER
    # above, so switching models and switching providers are both just env
    # vars: no code change needed for either.
    MODEL_DIR: str = os.environ.get(
        "LUCY_MODEL_DIR",
        r"C:\Users\tepui_66mr92q\trading_ai\models\qwen3_8b",
    )

settings = Settings()
