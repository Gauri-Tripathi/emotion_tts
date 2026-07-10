"""Schema tests."""

from pathlib import Path

import pytest

from unitts.core.schemas import AudioFormat, Emotion, ProviderCapabilities, SpeakingStyle, TTSRequest, TTSResponse, VoiceInfo


def test_tts_request_accepts_reference_audio_string() -> None:
    request = TTSRequest(text="hello", reference_audio="voice.wav", emotion="happy")
    assert request.reference_audio == [Path("voice.wav")]
    assert request.emotion is Emotion.HAPPY


def test_tts_request_requires_input() -> None:
    with pytest.raises(ValueError):
        TTSRequest()


def test_requested_schema_compatibility_fields() -> None:
    request = TTSRequest(text="hello", emotion="empathetic", speaking_style="customer-service")
    assert request.emotion is Emotion.EMPATHETIC
    assert request.speaking_style is SpeakingStyle.CUSTOMER_SERVICE
    assert request.sample_rate == 24000
    assert request.normalize is True
    assert request.trim_silence is True

    response = TTSResponse(audio=b"abc", metadata={"provider": "test", "voice": "voice-a"})
    assert response.audio_bytes == b"abc"
    assert response.format is AudioFormat.WAV
    assert response.provider == "test"
    assert response.voice_used == "voice-a"

    voice = VoiceInfo(id="v1", name="Voice", language=["en"])
    caps = ProviderCapabilities(name="provider", provider_type="gpu", supported_formats=[AudioFormat.WAV])
    assert voice.supports_cloning is False
    assert caps.requires_gpu is False
