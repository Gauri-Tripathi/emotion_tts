from __future__ import annotations

from collections.abc import AsyncIterator

from unitts.contracts import (
    AudioChunk,
    AudioResult,
    EngineHealth,
    EngineLoadError,
    EngineUnavailableError,
    InferenceContext,
    LoadContext,
    ReplicaState,
    SynthesisRequest,
)
from unitts.engine_sdk import SpeechEngine


class EngineWorker:
    """Lifecycle boundary used in-process for tests and behind IPC in production."""

    def __init__(self, engine: SpeechEngine) -> None:
        self.engine = engine
        self.state = ReplicaState.UNLOADED

    async def load(self, context: LoadContext) -> None:
        self.engine.manifest().assert_protocol(context.protocol_version)
        self.state = ReplicaState.LOADING
        try:
            await self.engine.load(context)
        except Exception as exc:
            self.state = ReplicaState.FAILED
            raise EngineLoadError(str(exc)) from exc
        self.state = ReplicaState.WARM

    async def generate(self, request: SynthesisRequest, context: InferenceContext) -> AudioResult:
        if self.state not in {ReplicaState.WARM, ReplicaState.ACTIVE}:
            raise EngineUnavailableError(f"Worker is {self.state}")
        self.state = ReplicaState.ACTIVE
        try:
            return await self.engine.generate(request, context)
        except Exception:
            # A request failure is isolated to the worker boundary. The process manager
            # may inspect health and restart without crashing the API process.
            health = await self.engine.health()
            if not health.healthy:
                self.state = ReplicaState.FAILED
            raise
        finally:
            if self.state is ReplicaState.ACTIVE:
                self.state = ReplicaState.WARM

    async def stream(self, request: SynthesisRequest, context: InferenceContext) -> AsyncIterator[AudioChunk]:
        if self.state is not ReplicaState.WARM:
            raise EngineUnavailableError(f"Worker is {self.state}")
        self.state = ReplicaState.ACTIVE
        try:
            async for chunk in self.engine.stream(request, context):
                yield chunk
        finally:
            if self.state is ReplicaState.ACTIVE:
                self.state = ReplicaState.WARM

    async def unload(self) -> None:
        await self.engine.unload()
        self.state = ReplicaState.UNLOADED

    async def health(self) -> EngineHealth:
        health = await self.engine.health()
        return health.model_copy(update={"state": self.state})
