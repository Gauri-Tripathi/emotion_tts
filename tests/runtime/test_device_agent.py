import pytest

from unitts.contracts import ReplicaState, ResourceEstimate
from unitts.runtime.device_agent import DeviceAgent, GPUDevice, Replica


def test_memory_admission_respects_safety_margin_and_peak_load() -> None:
    agent = DeviceAgent([GPUDevice("gpu0", total_memory_mb=10_000, fragmentation_reserve_mb=1000)])
    assert agent.can_admit("gpu0", ResourceEstimate(gpu_memory_mb=7000), peak_load_mb=2000)
    assert not agent.can_admit("gpu0", ResourceEstimate(gpu_memory_mb=7001), peak_load_mb=2000)


def test_replica_transitions_and_draining_before_eviction() -> None:
    agent = DeviceAgent([GPUDevice("gpu0", 10_000)])
    replica = Replica("r1", "dummy", "gpu0", 100, state=ReplicaState.WARM)
    agent.replicas[replica.replica_id] = replica
    agent.transition("r1", ReplicaState.ACTIVE)
    agent.transition("r1", ReplicaState.DRAINING)
    agent.request_eviction("r1")
    assert replica.state is ReplicaState.EVICTING

    other = Replica("r2", "dummy", "gpu0", 100, state=ReplicaState.WARM)
    agent.replicas[other.replica_id] = other
    with pytest.raises(ValueError, match="drained"):
        agent.request_eviction("r2")
