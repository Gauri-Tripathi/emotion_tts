"""Current open-weight model providers.

Older GPU defaults from the original prompt such as XTTS v2, Parler large, and
StyleTTS 2 are intentionally not registered as first-class defaults. They can
still be added externally through the registry, but UniTTS keeps the built-in
open-weight path focused on maintained providers with active ecosystems.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from unitts.core.capabilities import CapabilityMatrix, ProviderType
from unitts.core.device import cleanup_torch
from unitts.core.registry import ProviderRegistry
from unitts.core.schemas import TTSRequest, TTSResponse
from unitts.providers.base_http import HttpProvider


@ProviderRegistry.register("kokoro")
class KokoroProvider(HttpProvider):
    """Kokoro local open-weight TTS provider."""

    name = "kokoro"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.CPU,
        languages=["en", "ja", "zh", "es", "fr", "hi", "it", "pt"],
        speed=True,
        output_formats=["wav"],
        notes="Uses the installed kokoro package API.",
    )

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self._pipeline: Any | None = None

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        try:
            import soundfile as sf
            from kokoro import KPipeline
        except Exception as exc:
            raise ImportError("Install kokoro plus unitts[audio] to use the Kokoro provider") from exc
        lang_code = self.options.get("lang_code") or (request.language or "en")[:1].lower()
        if self._pipeline is None:
            self._pipeline = KPipeline(lang_code=lang_code)
        generator = self._pipeline(request.text or request.ssml or request.phonemes or "", voice=request.voice or self.options.get("voice", "af_heart"), speed=request.speed or 1.0)
        _graphemes, _phonemes, audio = next(generator)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output = Path(handle.name)
        try:
            sf.write(output, audio, request.sample_rate or 24000)
            data = output.read_bytes()
        finally:
            output.unlink(missing_ok=True)
        return TTSResponse(audio=data, sample_rate=request.sample_rate or 24000, metadata={"provider": self.name}, warnings=warnings)

    def cleanup(self) -> None:
        self._pipeline = None
        cleanup_torch()


@ProviderRegistry.register("f5")
@ProviderRegistry.register("f5-tts")
class F5TTSProvider(HttpProvider):
    """F5-TTS open-weight voice cloning provider."""

    name = "f5-tts"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.GPU,
        languages=["en", "zh", "many"],
        voice_cloning=True,
        speed=True,
        output_formats=["wav"],
        generation_controls=True,
        notes="Requires the F5-TTS package; reference_text is needed for best cloning.",
    )

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self._infer: Any | None = None

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        if not request.reference_audio:
            warnings.append("F5-TTS voice cloning works best with reference_audio")
        try:
            from f5_tts.api import F5TTS
        except Exception as exc:
            raise ImportError("Install F5-TTS and unitts[gpu] to use the F5-TTS provider") from exc
        if self._infer is None:
            self._infer = F5TTS(model=self.options.get("model", "F5TTS_v1_Base"), device=self.device)
        wav, sample_rate, _spect = self._infer.infer(
            ref_file=str(request.reference_audio[0]) if request.reference_audio else None,
            ref_text=request.reference_text or "",
            gen_text=request.text or request.ssml or request.phonemes or "",
            seed=request.seed,
        )
        return _write_float_wav(wav, sample_rate, {"provider": self.name}, warnings)

    def cleanup(self) -> None:
        self._infer = None
        cleanup_torch()


@ProviderRegistry.register("dia")
class DiaProvider(HttpProvider):
    """Dia open-weight dialogue and non-verbal speech provider."""

    name = "dia"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.GPU,
        languages=["en"],
        multi_speaker=True,
        dialogue=True,
        output_formats=["wav"],
        generation_controls=True,
        notes="Supports [S1]/[S2] dialogue markup and non-verbal tokens such as [laugh].",
    )

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self._model: Any | None = None

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        try:
            from dia.model import Dia
        except Exception as exc:
            raise ImportError("Install the Dia package and unitts[gpu] to use the Dia provider") from exc
        if self._model is None:
            self._model = Dia.from_pretrained(self.options.get("model", "nari-labs/Dia-1.6B"), device=self.device)
        audio = self._model.generate(
            request.text or request.ssml or request.phonemes or "",
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 0.95,
            cfg_scale=request.cfg_scale or 3.0,
        )
        return _write_float_wav(audio, request.sample_rate or 44100, {"provider": self.name}, warnings)

    def cleanup(self) -> None:
        self._model = None
        cleanup_torch()


def _write_float_wav(audio: Any, sample_rate: int, metadata: dict[str, Any], warnings: list[str]) -> TTSResponse:
    try:
        import soundfile as sf
    except Exception as exc:
        raise ImportError("Install unitts[audio] for local open-weight audio output") from exc
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        output = Path(handle.name)
    try:
        sf.write(output, audio, sample_rate)
        data = output.read_bytes()
    finally:
        output.unlink(missing_ok=True)
    return TTSResponse(audio=data, sample_rate=sample_rate, metadata=metadata, warnings=warnings)
