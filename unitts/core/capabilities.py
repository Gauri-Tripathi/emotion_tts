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
    voice_description: bool = False
    non_verbal: bool = False
    breathing: bool = False
    requires_api_key: bool = False
    requires_gpu: bool = False
    min_gpu_vram_gb: float | None = None
    max_text_length: int | None = None
    notes: str | None = None

    def unsupported_request_fields(self, request: object) -> list[str]:
        """Return request feature names that this provider will ignore."""

        unsupported: list[str] = []
        checks = {
            "ssml": (self.ssml, None),
            "reference_audio": (self.voice_cloning, []),
            "emotion": (self.emotion, None),
            "speaking_style": (self.speaking_style, None),
            "voice_description": (self.voice_description, None),
            "add_breathing": (self.breathing, False),
            "laughter": (self.non_verbal, False),
            "non_verbal_sounds": (self.non_verbal, []),
            "speed": (self.speed, 1.0),
            "pitch": (self.pitch, 0.0),
            "volume": (self.volume, 1.0),
            "stream": (self.streaming, False),
        }
        for field_name, supported in checks.items():
            supported, default = supported
            value = getattr(request, field_name, None)
            if value not in (None, False, [], "", default) and not supported:
                unsupported.append(field_name)
        return unsupported
