"""Provider capability declarations."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ProviderType(StrEnum):
    """High-level provider runtime category."""

    API = "api"
    CPU = "cpu"
    GPU = "gpu"


class CapabilityMatrix(BaseModel):
    """Feature support advertised by a provider."""

    provider_type: ProviderType
    languages: list[str] = Field(default_factory=list)
    voices: bool = True
    voice_cloning: bool = False
    streaming: bool = False
    ssml: bool = False
    emotion: bool = False
    speaking_style: bool = False
    speed: bool = False
    pitch: bool = False
    volume: bool = False
    multi_speaker: bool = False
    dialogue: bool = False
    output_formats: list[str] = Field(default_factory=lambda: ["wav"])
    generation_controls: bool = False
    notes: str | None = None

    def unsupported_request_fields(self, request: object) -> list[str]:
        """Return request feature names that this provider will ignore."""

        unsupported: list[str] = []
        checks = {
            "ssml": self.ssml,
            "reference_audio": self.voice_cloning,
            "emotion": self.emotion,
            "speaking_style": self.speaking_style,
            "speed": self.speed,
            "pitch": self.pitch,
            "volume": self.volume,
            "stream": self.streaming,
        }
        for field_name, supported in checks.items():
            value = getattr(request, field_name, None)
            if value not in (None, False, [], "") and not supported:
                unsupported.append(field_name)
        return unsupported
