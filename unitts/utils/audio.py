"""Small audio utilities used by providers."""

from __future__ import annotations

import io
import wave

from unitts.core.schemas import TTSResponse


def merge_wav_responses(responses: list[TTSResponse], pause_seconds: float = 0.0, warnings: list[str] | None = None) -> TTSResponse:
    """Merge PCM WAV responses into a single WAV response."""

    if not responses:
        return TTSResponse(audio=b"", warnings=warnings or [])

    params: wave._wave_params | None = None
    frames: list[bytes] = []
    for response in responses:
        with wave.open(io.BytesIO(response.audio), "rb") as reader:
            current = reader.getparams()
            if params is None:
                params = current
            elif current[:3] != params[:3]:
                raise ValueError("Cannot merge WAV files with different channel/sample-width/frame-rate")
            frames.append(reader.readframes(reader.getnframes()))

    assert params is not None
    pause = b"\x00" * int(params.framerate * pause_seconds) * params.nchannels * params.sampwidth
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams(params)
        for index, frame in enumerate(frames):
            if index:
                writer.writeframes(pause)
            writer.writeframes(frame)
    return TTSResponse(audio=output.getvalue(), sample_rate=params.framerate, warnings=warnings or [])


def normalize_wav_bytes(audio: bytes) -> bytes:
    """Normalize WAV audio when numpy/soundfile are available."""

    try:
        import numpy as np
        import soundfile as sf
    except Exception:
        return audio

    data, sample_rate = sf.read(io.BytesIO(audio), always_2d=False)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 0:
        return audio
    normalized = data / peak * 0.95
    output = io.BytesIO()
    sf.write(output, normalized, sample_rate, format="WAV")
    return output.getvalue()
