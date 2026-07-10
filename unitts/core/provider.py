"""Base provider contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from unitts.core.capabilities import CapabilityMatrix
from unitts.core.schemas import DialogueRequest, TTSRequest, TTSResponse


class BaseProvider(ABC):
    """Abstract base class for all TTS providers."""

    name: str
    capabilities: CapabilityMatrix

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        self.device = device
        self.options = kwargs

    @abstractmethod
    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Generate speech audio for a request."""

    def synthesize_dialogue(self, request: DialogueRequest) -> TTSResponse:
        """Generate a dialogue by concatenating individual turns."""

        from unitts.utils.audio import merge_wav_responses

        responses: list[TTSResponse] = []
        warnings: list[str] = []
        for turn in request.turns:
            voice = request.speaker_map.get(turn.speaker, turn.speaker)
            response = self.synthesize(TTSRequest(text=turn.text, voice=voice, emotion=turn.emotion))
            warnings.extend(response.warnings)
            responses.append(response)
        return merge_wav_responses(responses, pause_seconds=request.pause_between_turns, warnings=warnings)

    def list_voices(self) -> list[dict[str, Any]]:
        """Return voices known by the provider."""

        return []

    def cleanup(self) -> None:
        """Release provider resources."""

