import io
import wave

import pytest

from unitts.contracts import SynthesisRequest
from unitts.engines.dummy import DummyEngine
from unitts.runtime import UniTTSRuntime
from unitts.runtime.registry import EngineRegistry


def runtime() -> UniTTSRuntime:
    registry = EngineRegistry()
    registry.register_factory("dummy", DummyEngine)
    return UniTTSRuntime(registry)


@pytest.mark.asyncio
async def test_dummy_generation_vertical_slice_is_deterministic() -> None:
    service = runtime()
    request = SynthesisRequest(request_id="fixed", text="hello", engine_id="dummy", seed=7)
    first = await service.generate(request)
    second = await service.generate(request)
    assert first.data == second.data
    with wave.open(io.BytesIO(first.data), "rb") as reader:
        assert reader.getframerate() == 16000
        assert reader.getnframes() > 0
    await service.close()


@pytest.mark.asyncio
async def test_dummy_stream_ordering_and_final_marker() -> None:
    service = runtime()
    chunks = [chunk async for chunk in service.stream(SynthesisRequest(text="stream me", engine_id="dummy", streaming=True))]
    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.start_sample for chunk in chunks] == sorted(chunk.start_sample for chunk in chunks)
    assert chunks[-1].final is True
    assert all(not chunk.final for chunk in chunks[:-1])
    await service.close()
