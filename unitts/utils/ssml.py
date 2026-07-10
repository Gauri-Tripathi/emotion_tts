"""SSML construction helpers."""

from __future__ import annotations

import html


def simple_ssml(text: str, *, voice: str, language: str = "en-US") -> str:
    """Build simple SSML for providers that accept W3C SSML."""

    return f'<speak version="1.0" xml:lang="{language}"><voice name="{html.escape(voice)}">{html.escape(text)}</voice></speak>'
