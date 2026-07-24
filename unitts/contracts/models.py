"""Pydantic models used on the versioned engine-worker wire protocol."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROTOCOL_VERSION = "1.0"


class AudioFormat(StrEnum):
    WAV = "wav"
    PCM_S16LE = "pcm_s16le"
    MP3 = "mp3"
    FLAC = "flac"
    OPUS = "opus"


class AudioEncoding(StrEnum):
    WAV = "wav"
    PCM_S16LE = "pcm_s16le"
    MP3 = "mp3"
    FLAC = "flac"
    OPUS = "opus"


class GenerationMode(StrEnum):
    REALTIME = "realtime"
    ASYNCHRONOUS = "asynchronous"


class EngineStatus(StrEnum):
    PLANNED = "planned"
    EXPERIMENTAL = "experimental"
    READY = "ready"
    RESEARCH_ONLY = "research_only"
    LICENSE_BLOCKED = "license_blocked"
    DEPRECATED = "deprecated"


class ReplicaState(StrEnum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    WARM = "WARM"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    EVICTING = "EVICTING"
    FAILED = "FAILED"


class VoiceReference(BaseModel):
    uri: str
    transcript: str | None = None
    content_type: str | None = None
    sha256: str | None = None


class SynthesisRequest(BaseModel):
    """Architecture-neutral request; engine controls remain namespaced."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(min_length=1)
    engine_id: str | None = None
    language: str | None = None
    voice: str | None = None
    reference: VoiceReference | None = None
    speed: float = Field(default=1.0, gt=0.0, le=5.0)
    seed: int | None = None
    output_format: AudioFormat = AudioFormat.WAV
    streaming: bool = False
    mode: GenerationMode = GenerationMode.REALTIME
    controls: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class AudioResult(BaseModel):
    request_id: str
    data: bytes
    sample_rate: int = Field(gt=0)
    encoding: AudioEncoding
    duration_seconds: float = Field(ge=0.0)
    engine_id: str
    artifact_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioChunk(BaseModel):
    request_id: str
    sequence: int = Field(ge=0)
    start_sample: int = Field(ge=0)
    sample_rate: int = Field(gt=0)
    encoding: AudioEncoding
    data: bytes
    final: bool = False


class EngineCapabilities(BaseModel):
    streaming: bool = False
    voice_cloning: bool = False
    languages: list[str] = Field(default_factory=list)
    output_formats: list[AudioFormat] = Field(default_factory=lambda: [AudioFormat.WAV])
    execution_strategy: Literal["autoregressive", "flow_matching", "remote", "simple"] = "simple"
    supports_seed: bool = True
    safe_pause_boundaries: bool = False


class LicenseInfo(BaseModel):
    code: str
    weights: str
    commercial_use: Literal["allowed", "restricted", "verify", "unknown"] = "unknown"
    notes: str | None = None


class HardwareRequirements(BaseModel):
    accelerator: Literal["cpu", "cuda", "remote", "any"] = "any"
    min_vram_gb: float = Field(default=0.0, ge=0.0)
    min_driver: str | None = None
    cuda_runtime: str | None = None
    compute_capability: str | None = None
    peak_load_memory_mb: int = Field(default=0, ge=0)


class EngineManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    engine_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    family: str
    engine_version: str
    protocol_versions: list[str]
    status: EngineStatus
    entrypoint: str | None = None
    upstream_code: str
    checkpoints: list[str] = Field(default_factory=list)
    capabilities: EngineCapabilities
    hardware: HardwareRequirements = Field(default_factory=HardwareRequirements)
    license: LicenseInfo
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_supported_protocol(self) -> "EngineManifest":
        if not self.protocol_versions:
            raise ValueError("at least one protocol version is required")
        if self.status is EngineStatus.READY and not self.entrypoint:
            raise ValueError("ready engines require an entrypoint")
        return self

    def assert_protocol(self, protocol_version: str) -> None:
        from unitts.contracts.errors import ProtocolVersionError

        if protocol_version not in self.protocol_versions:
            raise ProtocolVersionError(
                f"Engine {self.engine_id} does not support protocol {protocol_version}",
                details={"supported": self.protocol_versions},
            )


class ResourceEstimate(BaseModel):
    gpu_memory_mb: int = Field(default=0, ge=0)
    host_memory_mb: int = Field(default=0, ge=0)
    estimated_seconds: float = Field(default=0.0, ge=0.0)
    token_count: int | None = Field(default=None, ge=0)
    frame_count: int | None = Field(default=None, ge=0)


class ExecutionPlan(BaseModel):
    request_id: str
    engine_id: str
    replica_id: str
    strategy: str
    tenant_id: str
    priority: int = 0
    protocol_version: str = PROTOCOL_VERSION
    resource_estimate: ResourceEstimate


class LoadContext(BaseModel):
    replica_id: str
    device: str = "cpu"
    protocol_version: str = PROTOCOL_VERSION
    model_cache: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class InferenceContext(BaseModel):
    request_id: str
    tenant_id: str = "local"
    deadline_monotonic: float | None = None
    trace_id: str | None = None


class EngineHealth(BaseModel):
    healthy: bool
    state: ReplicaState
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
