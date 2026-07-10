"""Open-weight local model providers.

All heavy model packages are imported lazily inside provider methods.
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
        requires_gpu=True,
        min_gpu_vram_gb=8.0,
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
        non_verbal=True,
        requires_gpu=True,
        min_gpu_vram_gb=10.0,
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


@ProviderRegistry.register("xtts")
@ProviderRegistry.register("xtts-v2")
class XTTSV2Provider(HttpProvider):
    """Coqui XTTS v2 provider."""

    name = "xtts-v2"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.GPU,
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko", "hi"],
        voice_cloning=True,
        speed=True,
        output_formats=["wav"],
        requires_gpu=True,
        min_gpu_vram_gb=4.0,
        notes="Uses Coqui TTS model tts_models/multilingual/multi-dataset/xtts_v2.",
    )

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self._model: Any | None = None

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        if not request.reference_audio:
            warnings.append("XTTS v2 voice cloning expects reference_audio; using provider default speaker if configured")
        try:
            from TTS.api import TTS
        except Exception as exc:
            raise ImportError("Install the Coqui TTS package to use XTTS v2: python -m pip install TTS") from exc
        if self._model is None:
            model_name = self.options.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")
            self._model = TTS(model_name=model_name).to(self.device)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output = Path(handle.name)
        try:
            kwargs: dict[str, Any] = {
                "text": request.text or request.ssml or request.phonemes or "",
                "file_path": str(output),
                "language": request.language or self.options.get("language", "en"),
            }
            if request.reference_audio:
                kwargs["speaker_wav"] = str(request.reference_audio[0])
            elif request.voice:
                kwargs["speaker"] = request.voice
            self._model.tts_to_file(**kwargs)
            data = output.read_bytes()
        finally:
            output.unlink(missing_ok=True)
        return TTSResponse(audio=data, sample_rate=request.sample_rate or 24000, metadata={"provider": self.name, "voice": request.voice}, warnings=warnings)

    def cleanup(self) -> None:
        self._model = None
        cleanup_torch()


@ProviderRegistry.register("cosyvoice")
@ProviderRegistry.register("cosyvoice2")
class CosyVoiceProvider(HttpProvider):
    """CosyVoice/CosyVoice2 provider."""

    name = "cosyvoice"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.GPU,
        languages=["zh", "en", "ja", "ko", "yue"],
        voice_cloning=True,
        streaming=True,
        emotion=True,
        speaking_style=True,
        output_formats=["wav"],
        generation_controls=True,
        requires_gpu=True,
        min_gpu_vram_gb=6.0,
        notes="Uses FunAudioLLM CosyVoice package APIs when installed.",
    )

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self._model: Any | None = None

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
        except Exception as exc:
            raise ImportError("Install FunAudioLLM CosyVoice to use this provider") from exc
        if self._model is None:
            model_path = self.options.get("model", "pretrained_models/CosyVoice2-0.5B")
            model_cls = CosyVoice2 if "2" in str(model_path).lower() else CosyVoice
            self._model = model_cls(model_path)

        text = request.text or request.ssml or request.phonemes or ""
        prompt_text = request.reference_text or self.options.get("prompt_text", "")
        prompt_audio = str(request.reference_audio[0]) if request.reference_audio else self.options.get("prompt_audio")
        if prompt_audio and hasattr(self._model, "inference_zero_shot"):
            results = self._model.inference_zero_shot(text, prompt_text, prompt_audio, stream=request.stream)
        elif hasattr(self._model, "inference_sft"):
            results = self._model.inference_sft(text, request.voice or self.options.get("voice", "中文女"), stream=request.stream)
        else:
            raise RuntimeError("Installed CosyVoice package does not expose a supported inference method")
        first = next(iter(results))
        audio = first.get("tts_speech") if isinstance(first, dict) else first
        sample_rate = int(self.options.get("sample_rate", request.sample_rate or 22050))
        return _write_float_wav(audio, sample_rate, {"provider": self.name, "voice": request.voice}, warnings)

    def cleanup(self) -> None:
        self._model = None
        cleanup_torch()


@ProviderRegistry.register("qwen3-tts")
@ProviderRegistry.register("qwen-tts")
class Qwen3TTSProvider(HttpProvider):
    """Qwen3-TTS provider.

    Qwen3-TTS is new and package APIs may vary. This provider supports a callable
    object with a ``generate`` method loaded from a configured Python package path.
    """

    name = "qwen3-tts"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.GPU,
        languages=["zh", "en", "de", "fr", "es", "it", "pt", "ja", "ko", "ar"],
        voice_cloning=True,
        streaming=True,
        voice_description=True,
        output_formats=["wav"],
        generation_controls=True,
        requires_gpu=True,
        min_gpu_vram_gb=8.0,
        notes="Qwen3-TTS supports short-reference cloning and description control; configure loader when package API changes.",
    )

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self._model: Any | None = None

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        if self._model is None:
            self._model = self._load_model()
        audio = self._call_inference(request)
        if isinstance(audio, tuple):
            audio, sample_rate = audio[:2]
        else:
            sample_rate = request.sample_rate or 24000
        return _write_float_wav(audio, int(sample_rate), {"provider": self.name, "voice": request.voice}, warnings)

    def _load_model(self) -> Any:
        import importlib

        module_name = self.options.get("module", "qwen3_tts")
        class_name = self.options.get("class_name", "Qwen3TTS")
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise ImportError(
                "Install the Qwen3-TTS package or pass module/class_name options for the installed implementation"
            ) from exc
        model_cls = getattr(module, class_name)
        if hasattr(model_cls, "from_pretrained"):
            model = model_cls.from_pretrained(self.options.get("model", "Qwen/Qwen3-TTS"))
        else:
            try:
                model = model_cls(model=self.options.get("model", "Qwen/Qwen3-TTS"), device=self.device)
            except TypeError:
                model = model_cls(model=self.options.get("model", "Qwen/Qwen3-TTS"))
        if self.device != "cpu" and hasattr(model, "to"):
            model = model.to(self.device)
        return model

    def _call_inference(self, request: TTSRequest) -> Any:
        text = request.text or request.ssml or request.phonemes or ""
        kwargs: dict[str, Any] = {
            "text": text,
            "input_text": text,
            "prompt": text,
            "voice": request.voice,
            "speaker": request.voice,
            "voice_description": request.voice_description,
            "reference_audio": [str(path) for path in request.reference_audio],
            "prompt_audio": str(request.reference_audio[0]) if request.reference_audio else None,
            "reference_text": request.reference_text,
            "prompt_text": request.reference_text,
            "language": request.language,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "seed": request.seed,
            "stream": request.stream,
        }
        for method_name in ("generate", "generate_speech", "synthesize", "tts", "infer", "inference", "__call__"):
            method = getattr(self._model, method_name, None)
            if method is None:
                continue
            try:
                return method(**_accepted_kwargs(method, kwargs))
            except TypeError as exc:
                try:
                    return method(text)
                except TypeError:
                    last_error = exc
                    continue
        public_methods = [
            name
            for name in dir(self._model)
            if not name.startswith("_") and callable(getattr(self._model, name, None))
        ]
        raise RuntimeError(
            "Configured Qwen3-TTS model object does not expose a supported inference method. "
            "Tried generate/generate_speech/synthesize/tts/infer/inference. "
            f"Available public methods include: {public_methods[:40]}"
        )

    def cleanup(self) -> None:
        self._model = None
        cleanup_torch()


def _write_float_wav(audio: Any, sample_rate: int, metadata: dict[str, Any], warnings: list[str]) -> TTSResponse:
    try:
        import soundfile as sf
    except Exception as exc:
        raise ImportError("Install unitts[audio] for local open-weight audio output") from exc
    import tempfile

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    elif hasattr(audio, "cpu") and hasattr(audio, "numpy"):
        audio = audio.cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        output = Path(handle.name)
    try:
        sf.write(output, audio, sample_rate)
        data = output.read_bytes()
    finally:
        output.unlink(missing_ok=True)
    return TTSResponse(audio=data, sample_rate=sample_rate, metadata=metadata, warnings=warnings)


def _accepted_kwargs(method: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    import inspect

    signature = inspect.signature(method)
    parameters = signature.parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in parameters
    }
