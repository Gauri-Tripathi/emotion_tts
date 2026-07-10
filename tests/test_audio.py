"""Audio utility tests."""

import io
import wave

from unitts.core.schemas import TTSResponse
from unitts.utils.audio import merge_wav_responses


def _wav(frames: bytes = b"\x00\x00\x00\x00") -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(10)
        writer.writeframes(frames)
    return output.getvalue()


def test_merge_wav_responses() -> None:
    merged = merge_wav_responses([TTSResponse(audio=_wav()), TTSResponse(audio=_wav())], pause_seconds=0.1)
    with wave.open(io.BytesIO(merged.audio), "rb") as reader:
        assert reader.getframerate() == 10
        assert reader.getnframes() == 5
