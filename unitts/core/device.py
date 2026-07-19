"""Device resolution helpers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto", min_vram_gb: float | None = None) -> str:
    """Resolve a requested device without importing torch unless needed."""

    if preference == "cpu":
        return preference
    if preference in {"cuda", "mps"}:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(f"{preference} was requested but PyTorch is not available") from exc
        if preference == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Run inside a GPU allocation and install "
                "a PyTorch CUDA build compatible with the NVIDIA driver."
            )
        if preference == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is unavailable")
        return preference
    if preference != "auto":
        raise ValueError("device must be one of auto, cpu, cuda, or mps")

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        if min_vram_gb is not None:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            if vram_gb < min_vram_gb:
                logger.warning("CUDA detected but %.1fGB VRAM is below %.1fGB", vram_gb, min_vram_gb)
                return "cpu"
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cleanup_torch() -> None:
    """Best-effort torch cleanup for GPU providers."""

    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return
