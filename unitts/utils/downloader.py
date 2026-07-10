"""Model download helpers."""

from __future__ import annotations

from pathlib import Path


def hf_download(repo_id: str, filename: str, cache_dir: str | Path | None = None) -> Path:
    """Download a file from Hugging Face Hub with local caching."""

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise ImportError("Install unitts[gpu] to download Hugging Face model files") from exc
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir) if cache_dir else None, resume_download=True))
