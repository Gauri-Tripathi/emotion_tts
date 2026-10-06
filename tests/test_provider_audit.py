"""Offline provider contract audit. No credentials or external requests required."""

import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import requests
import responses
import soundfile as sf
from click.testing import CliRunner

from unitts import TTSRequest
from unitts.cli import main
from unitts.core.registry import ProviderRegistry
from unitts.providers.api import (
    AzureProvider, ElevenLabsProvider, FishAudioProvider, OpenAIProvider, SmallestAIProvider,
)
from unitts.providers.open_weights import KokoroProvider


REST = [
    (ElevenLabsProvider, "https://api.elevenlabs.io/v1/text-to-speech/test-voice", "mp3"),
    (FishAudioProvider, "https://api.fish.audio/v1/tts", "wav"),
    (SmallestAIProvider, "https://waves-api.smallest.ai/api/v1/lightning/get_speech", "wav"),
    (AzureProvider, "https://testregion.tts.speech.microsoft.com/cognitiveservices/v1", "wav"),
]


def provider(cls):
    result = cls(api_key="test-placeholder", region="testregion")
    if cls is AzureProvider:
        result._azure_token = Mock(return_value="test-token")
    return result


@pytest.mark.parametrize("cls", [OpenAIProvider, *(row[0] for row in REST)])
def test_missing_credentials(cls):
    with pytest.raises(ValueError, match="requires"):
        cls().synthesize(TTSRequest(text="hello"))


@pytest.mark.parametrize("cls,url,fmt", REST)
@responses.activate
def test_rest_success(cls, url, fmt):
    responses.post(url, body=b"audio-fixture", content_type="audio/" + fmt)
    result = provider(cls).synthesize(TTSRequest(text="Hello & goodbye", voice="test-voice", output_format=fmt))
    assert result.audio == b"audio-fixture"
    assert result.output_format == fmt
    body = responses.calls[0].request.body
    if cls is AzureProvider:
        assert "Hello &amp; goodbye" in body
    else:
        assert json.loads(body)["text"] == "Hello & goodbye"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500, 503])
@pytest.mark.parametrize("cls,url,fmt", REST)
@responses.activate
def test_rest_errors(cls, url, fmt, status):
    responses.post(url, status=status, json={"error": "test failure"})
    with pytest.raises(RuntimeError, match=f"HTTP {status}"):
        provider(cls).synthesize(TTSRequest(text="hello", voice="test-voice", output_format=fmt))


@pytest.mark.parametrize("cls,url,fmt", REST)
@responses.activate
def test_rest_timeout(cls, url, fmt):
    responses.post(url, body=requests.Timeout("test timeout"))
    with pytest.raises(requests.Timeout):
        provider(cls).synthesize(TTSRequest(text="hello", voice="test-voice", output_format=fmt))


@pytest.mark.parametrize("fmt", OpenAIProvider.capabilities.output_formats)
def test_openai_sdk_payload(monkeypatch, fmt):
    create = Mock(return_value=SimpleNamespace(read=lambda: b"audio-fixture"))
    factory = Mock(return_value=SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create))))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    result = OpenAIProvider(api_key="test-placeholder", model="test-model").synthesize(
        TTSRequest(text="hello", voice="alloy", speed=1.2, output_format=fmt)
    )
    assert result.audio == b"audio-fixture"
    create.assert_called_once_with(model="test-model", voice="alloy", input="hello", response_format=fmt, speed=1.2)


@pytest.mark.parametrize("cls,url,fmt", REST)
@responses.activate
def test_reject_empty_audio(cls, url, fmt):
    responses.post(url, body=b"", content_type="audio/" + fmt)
    with pytest.raises((ValueError, RuntimeError)):
        provider(cls).synthesize(TTSRequest(text="hello", voice="test-voice", output_format=fmt))


@pytest.mark.parametrize("cls,url,fmt", REST)
@responses.activate
def test_reject_json_instead_of_audio(cls, url, fmt):
    responses.post(url, json={"error": "unexpected JSON response"})
    with pytest.raises((ValueError, RuntimeError)):
        provider(cls).synthesize(TTSRequest(text="hello", voice="test-voice", output_format=fmt))


@responses.activate
def test_elevenlabs_speed_is_forwarded():
    responses.post(REST[0][1], body=b"audio")
    provider(ElevenLabsProvider).synthesize(TTSRequest(text="hello", voice="test-voice", speed=1.5))
    assert json.loads(responses.calls[0].request.body)["voice_settings"]["speed"] == 1.5


def test_kokoro_retains_all_chunks(monkeypatch):
    pipeline = Mock(return_value=iter([("one", "", np.ones(100)), ("two", "", np.ones(200))]))
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=Mock(return_value=pipeline)))
    result = KokoroProvider(lang_code="a").synthesize(TTSRequest(text="one\ntwo"))
    assert sf.info(io.BytesIO(result.audio)).frames == 300


@pytest.mark.parametrize("field,expected", [("lang_code", "a"), ("device", "cpu")])
def test_kokoro_passes_cpu_device_and_english_code(monkeypatch, field, expected):
    factory = Mock(return_value=Mock(return_value=iter([("one", "", np.ones(100))])))
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=factory))
    KokoroProvider(device="cpu").synthesize(TTSRequest(text="hello", language="en"))
    assert factory.call_args.kwargs.get(field) == expected


@pytest.mark.parametrize("alias,canonical", [("espeak-ng", "espeak"), ("fish-audio", "fish"), ("smallest-ai", "smallest")])
def test_provider_aliases(alias, canonical):
    assert ProviderRegistry.get(alias) is ProviderRegistry.get(canonical)


@pytest.mark.parametrize("name", ["espeak", "piper", "kokoro", "openai", "elevenlabs", "fish", "smallest", "azure"])
def test_cli_rejects_empty_text(name):
    result = CliRunner().invoke(main, ["synthesize", "   ", "--provider", name])
    assert result.exit_code == 2
    assert "Provide text" in result.output


@pytest.mark.parametrize("field,value", [("speed", 0), ("speed", 6), ("volume", -1), ("volume", 3), ("pitch", 21), ("sample_rate", 0)])
def test_request_bounds(field, value):
    with pytest.raises(ValueError):
        TTSRequest(text="hello", **{field: value})


@pytest.mark.parametrize("status", [401, 403, 429, 500])
@responses.activate
def test_azure_token_failure(status):
    responses.post("https://testregion.api.cognitive.microsoft.com/sts/v1.0/issueToken", status=status)
    with pytest.raises(requests.HTTPError):
        AzureProvider(api_key="test-placeholder", region="testregion").synthesize(TTSRequest(text="hello"))


@responses.activate
def test_elevenlabs_voice_catalog():
    responses.get("https://api.elevenlabs.io/v1/voices", json={"voices": [{"voice_id": "v1", "name": "Voice one"}]})
    assert provider(ElevenLabsProvider).list_voices() == [{"id": "v1", "name": "Voice one"}]


def test_kokoro_empty_generation_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=Mock(return_value=Mock(return_value=iter([])))))
    with pytest.raises((ValueError, RuntimeError), match="audio|empty"):
        KokoroProvider(lang_code="a").synthesize(TTSRequest(text="..."))


def test_kokoro_sample_rate_preserves_duration(monkeypatch):
    pipeline = Mock(return_value=iter([("one", "", np.ones(240))]))
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=Mock(return_value=pipeline)))
    result = KokoroProvider(lang_code="a").synthesize(TTSRequest(text="one", sample_rate=48000))
    assert sf.info(io.BytesIO(result.audio)).duration == pytest.approx(0.01)


def test_openai_rejects_empty_audio(monkeypatch):
    create = Mock(return_value=SimpleNamespace(read=lambda: b""))
    factory = Mock(return_value=SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create))))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    with pytest.raises((ValueError, RuntimeError)):
        OpenAIProvider(api_key="test-placeholder").synthesize(TTSRequest(text="hello"))


def test_openai_cli_quota_error_is_readable(monkeypatch):
    import httpx
    from openai import RateLimitError

    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/audio/speech"))
    error = RateLimitError("Quota exceeded", response=response, body={"code": "insufficient_quota"})
    monkeypatch.setattr("unitts.core.engine.provider_options", lambda *args: {"api_key": "test-placeholder"})
    monkeypatch.setattr(OpenAIProvider, "synthesize", Mock(side_effect=error))
    result = CliRunner().invoke(main, ["synthesize", "hello", "--provider", "openai"])
    assert result.exit_code == 1
    assert "quota" in result.output.lower()
