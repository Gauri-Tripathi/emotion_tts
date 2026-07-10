"""Provider package.

Importing this module registers lightweight provider classes only. Heavy model
libraries are imported lazily inside provider methods.
"""

from unitts.providers.api import AzureProvider, ElevenLabsProvider, FishAudioProvider, OpenAIProvider, SmallestAIProvider
from unitts.providers.local import EspeakProvider, PiperProvider
from unitts.providers.open_weights import CosyVoiceProvider, DiaProvider, F5TTSProvider, KokoroProvider, Qwen3TTSProvider, XTTSV2Provider

__all__ = [
    "AzureProvider",
    "CosyVoiceProvider",
    "DiaProvider",
    "ElevenLabsProvider",
    "EspeakProvider",
    "F5TTSProvider",
    "FishAudioProvider",
    "KokoroProvider",
    "OpenAIProvider",
    "PiperProvider",
    "Qwen3TTSProvider",
    "SmallestAIProvider",
    "XTTSV2Provider",
]
