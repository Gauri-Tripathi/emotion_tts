"""Stable, lightweight wire contracts for the UniTTS engine protocol."""

from unitts.contracts.errors import (
    AdmissionError,
    EngineError,
    EngineLoadError,
    EngineUnavailableError,
    ProtocolVersionError,
    StreamBackpressureError,
    UnsupportedCapabilityError,
)
from unitts.contracts.models import (
    PROTOCOL_VERSION,
    AudioChunk,
    AudioEncoding,
    AudioFormat,
    AudioResult,
    EngineCapabilities,
    EngineHealth,
    EngineManifest,
    EngineStatus,
    ExecutionPlan,
    GenerationMode,
    InferenceContext,
    LicenseInfo,
    LoadContext,
    ReplicaState,
    ResourceEstimate,
    SynthesisRequest,
    VoiceReference,
)

__all__ = [
    "PROTOCOL_VERSION", "AdmissionError", "AudioChunk", "AudioEncoding",
    "AudioFormat", "AudioResult", "EngineCapabilities", "EngineError",
    "EngineHealth", "EngineLoadError", "EngineManifest", "EngineStatus",
    "EngineUnavailableError", "ExecutionPlan", "GenerationMode",
    "InferenceContext", "LicenseInfo", "LoadContext", "ProtocolVersionError",
    "ReplicaState", "ResourceEstimate", "StreamBackpressureError",
    "SynthesisRequest", "UnsupportedCapabilityError", "VoiceReference",
]
