from __future__ import annotations

from unitts.runtime.worker import EngineWorker


class WorkerManager:
    """Tracks isolated replicas; transports can replace local worker objects."""

    def __init__(self) -> None:
        self.workers: dict[str, EngineWorker] = {}

    def register(self, replica_id: str, worker: EngineWorker) -> None:
        if replica_id in self.workers:
            raise ValueError(f"Duplicate replica: {replica_id}")
        self.workers[replica_id] = worker

    async def terminate(self, replica_id: str) -> None:
        worker = self.workers.pop(replica_id)
        await worker.unload()
        # Production process supervisors terminate the replica process here; process
        # exit, rather than empty_cache(), is the definitive CUDA reclamation boundary.
