from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Protocol

from unitts.contracts import ReplicaState, ResourceEstimate


@dataclass
class GPUDevice:
    device_id: str
    total_memory_mb: int
    reserved_memory_mb: int = 0
    fragmentation_reserve_mb: int = 1024

    @property
    def available_memory_mb(self) -> int:
        return max(0, self.total_memory_mb - self.reserved_memory_mb - self.fragmentation_reserve_mb)


@dataclass
class Replica:
    replica_id: str
    engine_id: str
    device_id: str
    memory_mb: int
    state: ReplicaState = ReplicaState.UNLOADED
    last_used: float = field(default_factory=monotonic)
    predicted_demand: float = 0.0
    reload_cost: float = 0.0
    sla_penalty: float = 0.0


class EvictionPolicy(Protocol):
    def choose(self, replicas: list[Replica]) -> Replica | None: ...


class WeightedEvictionPolicy:
    def choose(self, replicas: list[Replica]) -> Replica | None:
        candidates = [r for r in replicas if r.state in {ReplicaState.WARM, ReplicaState.DRAINING}]
        if not candidates:
            return None
        now = monotonic()
        return max(candidates, key=lambda r: (now - r.last_used) + r.memory_mb / 1024 - r.predicted_demand * 10 - r.reload_cost - r.sla_penalty)


_TRANSITIONS = {
    ReplicaState.UNLOADED: {ReplicaState.LOADING, ReplicaState.FAILED},
    ReplicaState.LOADING: {ReplicaState.WARM, ReplicaState.FAILED},
    ReplicaState.WARM: {ReplicaState.ACTIVE, ReplicaState.DRAINING, ReplicaState.FAILED},
    ReplicaState.ACTIVE: {ReplicaState.WARM, ReplicaState.DRAINING, ReplicaState.FAILED},
    ReplicaState.DRAINING: {ReplicaState.EVICTING, ReplicaState.WARM, ReplicaState.FAILED},
    ReplicaState.EVICTING: {ReplicaState.UNLOADED, ReplicaState.FAILED},
    ReplicaState.FAILED: {ReplicaState.UNLOADED},
}


class DeviceAgent:
    """Sole deterministic placement authority for a single GPU host."""

    def __init__(self, devices: list[GPUDevice], *, allow_packing: bool = False, eviction_policy: EvictionPolicy | None = None) -> None:
        self.devices = {d.device_id: d for d in devices}
        self.replicas: dict[str, Replica] = {}
        self.allow_packing = allow_packing
        self.eviction_policy = eviction_policy or WeightedEvictionPolicy()

    def can_admit(self, device_id: str, estimate: ResourceEstimate, *, peak_load_mb: int = 0) -> bool:
        required = estimate.gpu_memory_mb + peak_load_mb
        device = self.devices[device_id]
        if not self.allow_packing and any(r.device_id == device_id and r.state is not ReplicaState.UNLOADED for r in self.replicas.values()):
            return False
        return required <= device.available_memory_mb

    def transition(self, replica_id: str, target: ReplicaState) -> None:
        replica = self.replicas[replica_id]
        if target not in _TRANSITIONS[replica.state]:
            raise ValueError(f"Invalid replica transition: {replica.state} -> {target}")
        replica.state = target
        replica.last_used = monotonic()

    def request_eviction(self, replica_id: str) -> None:
        replica = self.replicas[replica_id]
        if replica.state is not ReplicaState.DRAINING:
            raise ValueError("Replica must be drained before eviction")
        self.transition(replica_id, ReplicaState.EVICTING)

    def finish_eviction(self, replica_id: str) -> None:
        replica = self.replicas[replica_id]
        if replica.state is not ReplicaState.EVICTING:
            raise ValueError("Replica is not evicting")
        self.devices[replica.device_id].reserved_memory_mb -= replica.memory_mb
        self.transition(replica_id, ReplicaState.UNLOADED)
