from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass

from unitts.contracts import ResourceEstimate, SynthesisRequest


@dataclass(frozen=True)
class BatchItem:
    request: SynthesisRequest
    estimate: ResourceEstimate


class ExecutionStrategy(ABC):
    """Common strategy lifecycle; batching algorithms remain engine-owned."""

    @abstractmethod
    def compatibility_key(self, request: SynthesisRequest) -> Hashable: ...

    @abstractmethod
    def estimate(self, request: SynthesisRequest) -> ResourceEstimate: ...

    def form_batches(self, items: list[BatchItem], max_batch_size: int) -> list[list[BatchItem]]:
        groups: dict[Hashable, list[BatchItem]] = {}
        for item in items:
            groups.setdefault(self.compatibility_key(item.request), []).append(item)
        return [values[i:i + max_batch_size] for values in groups.values() for i in range(0, len(values), max_batch_size)]
