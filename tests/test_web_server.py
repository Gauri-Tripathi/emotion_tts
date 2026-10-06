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


def test_web_selects_mp3_for_elevenlabs_and_returns_actual_format(monkeypatch):
    from unitts.web_server import synthesize_payload

    def synthesize(provider, device, model, request):
        assert provider == "elevenlabs"
        assert request.output_format == "mp3"
        return TTSResponse(audio=b"fixture", output_format="mp3")

    monkeypatch.setattr("unitts.web_server.INFERENCE_SESSION.synthesize", synthesize)
    result = synthesize_payload({"text": "hello", "provider": "elevenlabs"})
    assert result["output_format"] == "mp3"


def test_web_rejects_unsupported_format_without_inference(monkeypatch):
    from unittest.mock import Mock
    from unitts.web_server import synthesize_payload

    mock = Mock()
    monkeypatch.setattr("unitts.web_server.INFERENCE_SESSION.synthesize", mock)
    with pytest.raises(ValueError, match="supports"):
        synthesize_payload({"text": "hello", "provider": "piper", "output_format": "mp3"})
    mock.assert_not_called()


def test_web_catalog_deduplicates_aliases_and_never_exposes_keys(monkeypatch):
    monkeypatch.setattr("unitts.web_server.provider_options", lambda *args: {"api_key": "private-test-value", "region": "eastus"})
    providers = provider_payloads()
    names = [item["name"] for item in providers]
    assert len(names) == len(set(names))
    assert "espeak-ng" not in names
    assert next(item for item in providers if item["name"] == "elevenlabs")["configured"]
    assert "private-test-value" not in str(providers)


def test_explicit_env_file_maps_existing_key_without_overriding_exports(tmp_path, monkeypatch):
    from unitts.core.config import load_env_file
    import os

    for key in ("UNITTS_ELEVENLABS_API_KEY", "ELEVEN_LAB_KEY"):
        monkeypatch.delenv(key, raising=False)
        # Ensure dotenv's new values are restored after the test.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    path = tmp_path / ".env"
    path.write_text("ELEVEN_LAB_KEY=local-fixture\n")
    load_env_file(path)
    assert os.environ["UNITTS_ELEVENLABS_API_KEY"] == "local-fixture"
    monkeypatch.setenv("UNITTS_ELEVENLABS_API_KEY", "exported-fixture")
    load_env_file(path)
    assert os.environ["UNITTS_ELEVENLABS_API_KEY"] == "exported-fixture"


@pytest.mark.parametrize("status,word", [(402, "payment"), (429, "Quota"), (401, "Authentication")])
def test_web_service_errors_do_not_forward_response_bodies(status, word):
    import requests
    from unitts.web_server import public_error

    response = requests.Response()
    response.status_code = status
    response._content = b"private-fixture-do-not-forward"
    cause = requests.HTTPError(response=response)
    error = RuntimeError("private-fixture-do-not-forward")
    error.__cause__ = cause
    message = public_error(error)
    assert word in message
    assert "private-fixture" not in message
