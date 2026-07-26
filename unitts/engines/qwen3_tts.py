from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from unitts.contracts import (
    AudioChunk,
    AudioEncoding,
    AudioResult,
    EngineCapabilities,
    EngineHealth,
    EngineManifest,
    EngineStatus,
    InferenceContext,
    LicenseInfo,
    LoadContext,
    ReplicaState,
    ResourceEstimate,
    SynthesisRequest,
)
from unitts.engine_sdk import SpeechEngine


class QwenControls(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float | None = None
    top_p: float | None = None
    voice_description: str | None = None


class Qwen3TTSEngine(SpeechEngine):
    """Isolated adapter around the repository's best-tested real integration."""

    controls_model = QwenControls

    def __init__(self) -> None:
        self._provider: Any | None = None
        self._load_context: LoadContext | None = None

    @classmethod
    def manifest(cls) -> EngineManifest:
        return EngineManifest(
            engine_id="qwen3-tts", family="Qwen3-TTS", engine_version="0.1.0",
            protocol_versions=["1.0"], status=EngineStatus.EXPERIMENTAL,
            entrypoint="unitts.engines.qwen3_tts:Qwen3TTSEngine",
            upstream_code="https://github.com/QwenLM/Qwen3-TTS",
            checkpoints=["Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"],
            capabilities=EngineCapabilities(streaming=False, voice_cloning=True, execution_strategy="autoregressive"),
            license=LicenseInfo(code="Apache-2.0 (verify upstream version)", weights="checkpoint-specific; verify", commercial_use="verify"),
            limitations=["Requires an isolated qwen-tts/Torch environment", "Streaming adapter is not yet validated"],
        )

    async def load(self, context: LoadContext) -> None:
        self.manifest().assert_protocol(context.protocol_version)
        from unitts.providers.open_weights import Qwen3TTSProvider
        self._provider = Qwen3TTSProvider(device=context.device, **context.settings)
        # Preserve lazy weight loading from the working provider.
        self._load_context = context

    async def generate(self, request: SynthesisRequest, context: InferenceContext) -> AudioResult:
        if self._provider is None:
            raise RuntimeError("Qwen3-TTS engine is not loaded")
        controls = QwenControls.model_validate(request.controls)
        from unitts.core.schemas import OutputFormat, TTSRequest
        legacy = TTSRequest(
            text=request.text, language=request.language, voice=request.voice,
            reference_audio=[Path(request.reference.uri)] if request.reference else [],
            reference_text=request.reference.transcript if request.reference else None,
            speed=request.speed, seed=request.seed, output_format=OutputFormat.WAV,
            voice_description=controls.voice_description,
            temperature=controls.temperature,
            top_p=controls.top_p,
        )
        response = await asyncio.to_thread(self._provider.synthesize, legacy)
        return AudioResult(
            request_id=request.request_id, data=response.audio, sample_rate=response.sample_rate or 24000,
            encoding=AudioEncoding.WAV, duration_seconds=response.duration_seconds,
            engine_id="qwen3-tts", metadata=response.metadata,
        )

    async def stream(self, request: SynthesisRequest, context: InferenceContext) -> AsyncIterator[AudioChunk]:
        from unitts.contracts import UnsupportedCapabilityError
        raise UnsupportedCapabilityError("Qwen3-TTS streaming has not been validated in this adapter")
        yield

    async def unload(self) -> None:
        if self._provider is not None:
            await asyncio.to_thread(self._provider.cleanup)
        self._provider = None

    def estimate_resources(self, request: SynthesisRequest) -> ResourceEstimate:
        return ResourceEstimate(gpu_memory_mb=8192, token_count=max(1, len(request.text.split()) * 3))

    async def health(self) -> EngineHealth:
        return EngineHealth(healthy=self._provider is not None, state=ReplicaState.WARM if self._provider else ReplicaState.UNLOADED)
