"""One-time download of BAAI/bge-small-en-v1.5 (ONNX export) for Lucy's RAG
embedder, into models/bge_small_en_v15/.

Downloads from Hugging Face (huggingface.co/BAAI/bge-small-en-v1.5):
    onnx/model.onnx   ~133 MB
    tokenizer.json    ~0.7 MB
    config.json       <1 KB

Uses huggingface_hub, already installed in .venv. Safe to re-run: files
already present are skipped.

Run from the project root:
    python download_bge_model.py
"""

from huggingface_hub import hf_hub_download

from lucy.rag.embedder import BGE_MODEL_DIR, BGE_MODEL_ID, bge_model_files_present

FILES = ["onnx/model.onnx", "tokenizer.json", "config.json"]


def main():
    BGE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        target = BGE_MODEL_DIR / filename
        if target.exists():
            print(f"already present: {target}")
            continue
        print(f"downloading {BGE_MODEL_ID}/{filename} ...")
        hf_hub_download(
            repo_id=BGE_MODEL_ID,
            filename=filename,
            local_dir=str(BGE_MODEL_DIR),
            local_dir_use_symlinks=False,
        )

    if bge_model_files_present():
        print(f"\nOK — {BGE_MODEL_ID} ready in {BGE_MODEL_DIR}")
        print("Next: python migrate_vectorstore_to_bge.py")
    else:
        raise SystemExit("Download finished but model files are missing — check the output above.")


if __name__ == "__main__":
    main()
