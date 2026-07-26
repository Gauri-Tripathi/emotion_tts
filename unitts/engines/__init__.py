"""First-party lightweight adapters; heavyweight dependencies stay in engine envs."""

from unitts.engines.dummy import DummyEngine
from unitts.engines.qwen3_tts import Qwen3TTSEngine

__all__ = ["DummyEngine", "Qwen3TTSEngine"]
