"""Schema tests."""

from pathlib import Path

import pytest

from unitts.core.schemas import Emotion, TTSRequest


def test_tts_request_accepts_reference_audio_string() -> None:
    request = TTSRequest(text="hello", reference_audio="voice.wav", emotion="happy")
    assert request.reference_audio == [Path("voice.wav")]
    assert request.emotion is Emotion.HAPPY


def test_tts_request_requires_input() -> None:
    with pytest.raises(ValueError):
        TTSRequest()
