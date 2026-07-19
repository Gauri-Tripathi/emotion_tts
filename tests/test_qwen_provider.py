import pytest

from unitts import TTSRequest
from unitts.providers.open_weights import Qwen3TTSProvider


class _CustomVoiceModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def generate_custom_voice(self, **kwargs: str) -> tuple[list[object], int]:
        self.calls.append(kwargs)
        return [object()], 24000


def test_qwen_custom_voice_uses_supported_default_speaker_and_language_name() -> None:
    provider = Qwen3TTSProvider(device="cpu")
    model = _CustomVoiceModel()
    provider._model = model

    provider._call_inference(TTSRequest(text="Hello", language="en-US"))

    assert model.calls == [{"text": "Hello", "speaker": "ryan", "language": "english"}]


def test_qwen_custom_voice_rejects_voice_cloning_before_model_call() -> None:
    provider = Qwen3TTSProvider(device="cpu")
    provider._model = _CustomVoiceModel()

    with pytest.raises(ValueError, match="does not support voice cloning"):
        provider._call_inference(TTSRequest(text="Hello", reference_audio="reference.wav"))


def test_qwen_custom_voice_rejects_voice_design_before_model_call() -> None:
    provider = Qwen3TTSProvider(device="cpu")
    provider._model = _CustomVoiceModel()

    with pytest.raises(ValueError, match="does not support description-driven"):
        provider._call_inference(TTSRequest(text="Hello", voice_description="warm voice"))
