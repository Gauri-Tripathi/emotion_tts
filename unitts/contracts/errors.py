"""Structured errors shared by clients, runtime, and engine workers."""

from __future__ import annotations

from typing import Any


class EngineError(Exception):
    """Base error that can be serialized at a process boundary."""

    code = "engine_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ProtocolVersionError(EngineError):
    code = "protocol_version_mismatch"


class UnsupportedCapabilityError(EngineError):
    code = "unsupported_capability"


class EngineLoadError(EngineError):
    code = "engine_load_failed"


class EngineUnavailableError(EngineError):
    code = "engine_unavailable"


class AdmissionError(EngineError):
    code = "admission_rejected"


class StreamBackpressureError(EngineError):
    code = "stream_backpressure"
