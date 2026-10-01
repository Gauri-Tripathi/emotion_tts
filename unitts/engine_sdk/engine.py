from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel

from unitts.contracts import (
    AudioChunk,
    AudioResult,
    EngineHealth,
    EngineManifest,
    InferenceContext,
    LoadContext,
    ResourceEstimate,
    SynthesisRequest,
)


class SpeechEngine(ABC):
    """The only model-specific abstraction exposed to UniTTS Core."""

    controls_model: type[BaseModel] | None = None

    @classmethod
    @abstractmethod
    def manifest(cls) -> EngineManifest: ...

    @abstractmethod
    async def load(self, context: LoadContext) -> None: ...

    @abstractmethod
    async def generate(self, request: SynthesisRequest, context: InferenceContext) -> AudioResult: ...

    @abstractmethod
    async def stream(self, request: SynthesisRequest, context: InferenceContext) -> AsyncIterator[AudioChunk]:
        if False:
            yield AudioChunk(request_id="", sequence=0, start_sample=0, sample_rate=1, encoding="pcm_s16le", data=b"")

    @abstractmethod
    async def unload(self) -> None: ...

    @abstractmethod
    def estimate_resources(self, request: SynthesisRequest) -> ResourceEstimate: ...

    @abstractmethod
    async def health(self) -> EngineHealth: ...

    def validate_controls(self, request: SynthesisRequest) -> BaseModel | dict[str, object]:
        if self.controls_model is None:
            if request.controls:
                from unitts.contracts import UnsupportedCapabilityError
                raise UnsupportedCapabilityError("This engine does not accept architecture-specific controls")
            return {}
        return self.controls_model.model_validate(request.controls)
