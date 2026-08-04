import pytest

from unitts import TTSRequest, TTSResponse
from unitts.web_server import InferenceSession, provider_payloads


def test_web_provider_payloads_are_available_without_loading_models() -> None:
    providers = provider_payloads()

    qwen = next(provider for provider in providers if provider["name"] == "qwen3-tts")
    assert qwen["capabilities"]["requires_gpu"] is True


def test_inference_session_reuses_model_and_releases_on_switch(monkeypatch):
    events = []

    class FakeTTS:
        def __init__(self, provider, device, **options):
            self.provider = provider
            events.append(("load", provider, options))

        def synthesize(self, request):
            return TTSResponse(audio=request.text.encode())

        def cleanup(self):
            events.append(("unload", self.provider))

    monkeypatch.setattr("unitts.web_server.UniTTS", FakeTTS)
    session = InferenceSession()
    request = TTSRequest(text="hello")
    for _ in range(2):
        assert session.synthesize("piper", "cpu", None, request).audio == b"hello"
    assert events == [("load", "piper", {})]
    session.synthesize("qwen3-tts", "cuda", "checkpoint", request)
    assert events[1:] == [("unload", "piper"), ("load", "qwen3-tts", {"model": "checkpoint"})]
    session.synthesize("qwen3-tts", "cuda", "another", request)
    assert events[-2:] == [("unload", "qwen3-tts"), ("load", "qwen3-tts", {"model": "another"})]
    session.close()
    session.close()
    assert events[-1] == ("unload", "qwen3-tts")


def test_inference_failure_discards_cached_model(monkeypatch):
    from unittest.mock import Mock

    failed = Mock()
    failed.synthesize.side_effect = RuntimeError("inference failed")
    healthy = Mock()
    factory = Mock(side_effect=[failed, healthy])
    monkeypatch.setattr("unitts.web_server.UniTTS", factory)
    session = InferenceSession()
    request = TTSRequest(text="hello")
    with pytest.raises(RuntimeError, match="inference failed"):
        session.synthesize("piper", "cpu", None, request)
    failed.cleanup.assert_called_once()
    session.synthesize("piper", "cpu", None, request)
    assert factory.call_count == 2
    session.close()
