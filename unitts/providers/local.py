"""Local CPU providers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from unitts.core.capabilities import CapabilityMatrix, ProviderType
from unitts.core.registry import ProviderRegistry
from unitts.core.schemas import TTSRequest, TTSResponse
from unitts.providers.base_http import HttpProvider


@ProviderRegistry.register("piper")
class PiperProvider(HttpProvider):
    """Piper local CPU provider."""

    name = "piper"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.CPU,
        languages=["many"],
        speed=True,
        volume=True,
        multi_speaker=True,
        output_formats=["wav"],
    )

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self._voice: Any | None = None
        self._voice_name: str | None = None

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        if request.output_format.value != "wav":
            warnings.append("Piper outputs wav; requested format was ignored")
        try:
            import numpy as np
            import soundfile as sf
            from piper.config import SynthesisConfig
            from piper.download_voices import download_voice
            from piper.voice import PiperVoice
        except Exception as exc:
            raise ImportError("Install unitts[cpu] to use Piper") from exc

        cache_dir = Path(self.options.get("model_cache", Path.home() / ".unitts" / "models" / "piper"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        voice_name = request.voice or self.options.get("voice", "en_US-lessac-medium")
        model_path = cache_dir / f"{voice_name}.onnx"
        config_path = cache_dir / f"{voice_name}.onnx.json"
        if not model_path.exists() or not config_path.exists():
            download_voice(voice_name, cache_dir)
        if self._voice is None or self._voice_name != voice_name:
            self._voice = PiperVoice.load(model_path, config_path=config_path, download_dir=cache_dir)
            self._voice_name = voice_name
        length_scale = None if request.speed is None else 1.0 / request.speed
        syn_config = SynthesisConfig(length_scale=length_scale, volume=request.volume or 1.0)
        chunks = list(self._voice.synthesize(request.text or request.ssml or request.phonemes or "", syn_config=syn_config))
        audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            sf.write(temp_path, audio, self._voice.config.sample_rate)
            data = temp_path.read_bytes()
        finally:
            temp_path.unlink(missing_ok=True)
        return TTSResponse(audio=data, sample_rate=self._voice.config.sample_rate, metadata={"provider": self.name, "voice": voice_name}, warnings=warnings)

    def list_voices(self) -> list[dict[str, Any]]:
        try:
            from piper.download_voices import get_voices
        except Exception:
            return []
        voices = get_voices(self.options.get("download_url"), self.options.get("voices_file"))
        return [{"id": key, **value} for key, value in voices.items()]

    def cleanup(self) -> None:
        self._voice = None


@ProviderRegistry.register("espeak")
@ProviderRegistry.register("espeak-ng")
class EspeakProvider(HttpProvider):
    """eSpeak-NG subprocess provider."""

    name = "espeak"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.CPU,
        languages=["many"],
        speed=True,
        pitch=True,
        volume=True,
        output_formats=["wav"],
    )

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        binary = self.options.get("binary") or shutil.which("espeak-ng") or shutil.which("espeak")
        if not binary:
            raise RuntimeError("eSpeak-NG is not installed or not on PATH")
        text = request.text or request.ssml or request.phonemes or ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output = Path(handle.name)
        cmd = [binary, "-w", str(output)]
        if request.language:
            cmd += ["-v", request.language]
        if request.speed:
            cmd += ["-s", str(int(175 * request.speed))]
        if request.pitch:
            cmd += ["-p", str(int(max(0, min(99, 50 + request.pitch * 2))))]
        if request.volume:
            cmd += ["-a", str(int(max(0, min(200, request.volume * 100))))]
        if request.word_gap:
            cmd += ["-g", str(int(request.word_gap * 100))]
        cmd.append(text)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            audio = output.read_bytes()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"eSpeak failed: {exc.stderr}") from exc
        finally:
            output.unlink(missing_ok=True)
        return TTSResponse(audio=audio, sample_rate=22050, metadata={"provider": self.name}, warnings=warnings)
