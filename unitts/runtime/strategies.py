from __future__ import annotations

from collections.abc import Hashable

from unitts.contracts import ResourceEstimate, SynthesisRequest
from unitts.engine_sdk import ExecutionStrategy


class AutoregressiveStrategy(ExecutionStrategy):
    """Groups codec-LM requests sharing decode/KV-cache constraints."""

    def compatibility_key(self, request: SynthesisRequest) -> Hashable:
        c = request.controls
        return (request.engine_id, c.get("codec"), c.get("temperature"), c.get("top_p"), request.language)

    def estimate(self, request: SynthesisRequest) -> ResourceEstimate:
        tokens = max(1, len(request.text.split()) * 3)
        return ResourceEstimate(token_count=tokens, gpu_memory_mb=256 + tokens * 2, estimated_seconds=tokens / 30)


class FlowMatchingStrategy(ExecutionStrategy):
    """Groups fixed sampling trajectories by duration, solver, and CFG expansion."""

    def compatibility_key(self, request: SynthesisRequest) -> Hashable:
        c = request.controls
        frames = int(c.get("duration_frames", max(1, len(request.text) * 6)))
        bucket = ((frames + 255) // 256) * 256
        return (request.engine_id, bucket, c.get("solver", "euler"), c.get("steps", 32), bool(c.get("cfg", False)))

    def estimate(self, request: SynthesisRequest) -> ResourceEstimate:
        frames = int(request.controls.get("duration_frames", max(1, len(request.text) * 6)))
        return ResourceEstimate(frame_count=frames, gpu_memory_mb=512 + frames, estimated_seconds=frames / 500)


class RemoteProviderStrategy(ExecutionStrategy):
    """Compatibility for provider concurrency/rate-limit lanes; no GPU estimate."""

    def compatibility_key(self, request: SynthesisRequest) -> Hashable:
        return (request.engine_id, request.controls.get("provider_region"), request.streaming)

    def estimate(self, request: SynthesisRequest) -> ResourceEstimate:
        return ResourceEstimate(host_memory_mb=16, estimated_seconds=max(0.1, len(request.text) / 100))


class SimpleStrategy(ExecutionStrategy):
    def compatibility_key(self, request: SynthesisRequest) -> Hashable:
        return request.engine_id

    def estimate(self, request: SynthesisRequest) -> ResourceEstimate:
        return ResourceEstimate(host_memory_mb=8, estimated_seconds=max(0.01, len(request.text) / 1000))


STRATEGIES: dict[str, type[ExecutionStrategy]] = {
    "autoregressive": AutoregressiveStrategy,
    "flow_matching": FlowMatchingStrategy,
    "remote": RemoteProviderStrategy,
    "simple": SimpleStrategy,
}
