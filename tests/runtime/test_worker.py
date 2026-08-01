import pytest

from unitts.contracts import InferenceContext, LoadContext, ReplicaState, SynthesisRequest
from unitts.engines.dummy import DummyEngine
from unitts.runtime.worker import EngineWorker


class FailingEngine(DummyEngine):
    async def generate(self, request: SynthesisRequest, context: InferenceContext):
        self.loaded = False
        raise RuntimeError("model process failure")


@pytest.mark.asyncio
async def test_worker_failure_is_isolated_and_marked_failed() -> None:
    worker = EngineWorker(FailingEngine())
    await worker.load(LoadContext(replica_id="failing"))
    with pytest.raises(RuntimeError, match="model process failure"):
        await worker.generate(SynthesisRequest(text="hello"), InferenceContext(request_id="r"))
    assert worker.state is ReplicaState.FAILED
