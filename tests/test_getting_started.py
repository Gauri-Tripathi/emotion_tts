"""Regression coverage for CPU and API entry points (no network or weights)."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import responses
from click.testing import CliRunner

from unitts import UniTTS
from unitts.cli import main
from unitts.core.config import init_config, load_config
from unitts.core.schemas import TTSResponse
from unitts.providers.api import ElevenLabsProvider
from unitts.providers.local import PiperProvider


@pytest.mark.parametrize("empty", [False, True])
def test_piper_load_and_synthesis_without_downloads(tmp_path, monkeypatch, empty):
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    import io
    import wave

    from unitts import TTSRequest

    for suffix in (".onnx", ".onnx.json"):
        (tmp_path / f"en_US-lessac-medium{suffix}").touch()

    def synthesize(text, syn_config):
        assert syn_config.volume == 0.0
        return iter([] if empty else [SimpleNamespace(audio_float_array=np.zeros(100, dtype=np.float32)) for _ in range(2)])

    def load(model_path, config_path):
        return SimpleNamespace(config=SimpleNamespace(sample_rate=22050), synthesize=synthesize)

    monkeypatch.setitem(sys.modules, "piper.config", SimpleNamespace(SynthesisConfig=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "piper.voice", SimpleNamespace(PiperVoice=SimpleNamespace(load=load)))
    monkeypatch.setitem(sys.modules, "piper.download_voices", SimpleNamespace(download_voice=Mock(side_effect=AssertionError("unexpected download"))))
    provider = PiperProvider(model_cache=tmp_path)
    if empty:
        with pytest.raises(RuntimeError, match="produced no audio"):
            provider.synthesize(TTSRequest(text="hello", volume=0.0))
    else:
        result = provider.synthesize(TTSRequest(text="hello", volume=0.0))
        with wave.open(io.BytesIO(result.audio)) as reader:
            assert reader.getframerate() == 22050
            assert reader.getnframes() == 200
        assert result.duration_seconds == 200 / 22050


@responses.activate
def test_piper_voice_list_uses_catalog_without_version_specific_helper(monkeypatch):
    catalog = "https://example.test/voices.json"
    monkeypatch.setitem(sys.modules, "piper.download_voices", SimpleNamespace(VOICES_JSON=catalog))
    responses.add(responses.GET, catalog, json={"en_US-lessac-medium": {"name": "lessac"}})
    assert PiperProvider().list_voices()[0]["id"] == "en_US-lessac-medium"


def test_config_init_is_portable_and_preserves_existing_file(tmp_path, monkeypatch):
    monkeypatch.delenv("USERPROFILE", raising=False)
    path = init_config(tmp_path / "config.yaml")
    assert "${" not in load_config(path)["defaults"]["model_cache"]
    path.write_text("providers: {}\n", encoding="utf-8")
    init_config(path)
    assert path.read_text(encoding="utf-8") == "providers: {}\n"


@pytest.mark.parametrize("provider", ["piper", "elevenlabs"])
def test_cpu_and_api_skip_gpu_detection(monkeypatch, provider):
    monkeypatch.setattr("unitts.core.engine.provider_options", lambda *args: {})
    resolve = Mock(return_value="cpu")
    monkeypatch.setattr("unitts.core.engine.resolve_device", resolve)
    with UniTTS(provider=provider):
        resolve.assert_called_once_with("cpu", min_vram_gb=None)


def test_cli_mp3_provider_selects_matching_default(monkeypatch):
    monkeypatch.setattr("unitts.core.engine.provider_options", lambda *args: {})
    def synthesize(self, request):
        assert request.output_format.value == "mp3"
        return TTSResponse(audio=b"test-audio", output_format="mp3")
    monkeypatch.setattr(ElevenLabsProvider, "synthesize", synthesize)
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["synthesize", "hello", "-p", "elevenlabs"])
        assert result.exit_code == 0, result.output
        assert "output.mp3" in result.output


@pytest.mark.parametrize("args", [
    ["-p", "elevenlabs", "-o", "hello.wav"],
    ["-p", "piper", "--format", "wav", "-o", "hello.mp3"],
])
def test_cli_rejects_mismatched_formats_before_request(args):
    result = CliRunner().invoke(main, ["synthesize", "hello", *args])
    assert result.exit_code == 2
    assert "Error:" in result.output


@pytest.mark.parametrize("command", [["synthesize", "hello"], ["voices"]])
def test_cli_missing_api_key_is_readable(monkeypatch, command):
    monkeypatch.setattr("unitts.core.engine.provider_options", lambda *args: {})
    result = CliRunner().invoke(main, [*command, "-p", "elevenlabs"])
    assert result.exit_code == 1
    assert "UNITTS_ELEVENLABS_API_KEY" in result.output
    assert "Traceback" not in result.output


@responses.activate
def test_api_authentication_failure_includes_actionable_hint():
    responses.add(responses.POST, "https://api.elevenlabs.io/v1/text-to-speech/voice-id", status=401, json={"detail": "Unauthorized"})
    from unitts import TTSRequest
    with pytest.raises(RuntimeError, match="Check your API key"):
        ElevenLabsProvider(api_key="test-key").synthesize(TTSRequest(text="hello", voice="voice-id"))


@responses.activate
def test_api_request_uses_provider_voice_and_key():
    responses.add(responses.POST, "https://api.elevenlabs.io/v1/text-to-speech/voice-id", body=b"mp3-audio", content_type="audio/mpeg")
    from unitts import TTSRequest
    result = ElevenLabsProvider(api_key="test-key").synthesize(TTSRequest(text="hello", voice="voice-id", output_format="mp3"))
    assert result.audio == b"mp3-audio"
    assert result.output_format.value == "mp3"
    assert responses.calls[0].request.headers["xi-api-key"] == "test-key"
