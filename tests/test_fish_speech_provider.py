import base64

from unitts import TTSRequest
from unitts.providers.open_weights import FishSpeechProvider, _fish_speech_audio


class _Response:
    def __init__(self, *, content: bytes, content_type: str, payload: object | None = None) -> None:
        self.content = content
        self.headers = {"content-type": content_type}
        self._payload = payload

    def json(self) -> object:
        return self._payload


def test_fish_speech_accepts_raw_audio() -> None:
    response = _Response(content=b"RIFF", content_type="audio/wav")

    assert _fish_speech_audio(response) == b"RIFF"


def test_fish_speech_accepts_base64_json_audio() -> None:
    response = _Response(
        content=b"",
        content_type="application/json",
        payload={"audio": base64.b64encode(b"RIFF").decode("ascii")},
    )

    assert _fish_speech_audio(response) == b"RIFF"


def test_fish_speech_is_registered_as_a_gpu_provider() -> None:
    provider = FishSpeechProvider(device="cpu")

    assert provider.name == "fish-speech"
    assert provider.capabilities.voice_cloning is True
    assert TTSRequest(text="Hello").language == "en"
