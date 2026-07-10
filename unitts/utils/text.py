"""Text normalization helpers."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Apply conservative whitespace and URL normalization."""

    text = re.sub(r"https?://\S+", " link ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
