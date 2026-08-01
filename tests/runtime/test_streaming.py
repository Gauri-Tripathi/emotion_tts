import asyncio

import pytest

from unitts.contracts import AudioChunk, GenerationMode, StreamBackpressureError
from unitts.runtime.streaming import BoundedAudioStream


def chunk(sequence: int, *, final: bool = False, size: int = 20) -> AudioChunk:
    return AudioChunk(request_id="r", sequence=sequence, start_sample=sequence * size // 2, sample_rate=1000, encoding="pcm_s16le", data=b"0" * size, final=final)


@pytest.mark.asyncio
async def test_bounded_queue_backpressure_cancels_slow_realtime_client() -> None:
    stream = BoundedAudioStream(sample_rate=1000, max_buffered_seconds=0.005, slow_client_timeout=0.001)
    with pytest.raises(StreamBackpressureError):
        await stream.put(chunk(0))
    assert stream.cancelled


@pytest.mark.asyncio
async def test_stream_cancellation_stops_consumer() -> None:
    stream = BoundedAudioStream(sample_rate=1000)
    task = asyncio.create_task(anext(stream.__aiter__()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    stream.cancel()
    assert stream.cancelled


@pytest.mark.asyncio
async def test_queue_never_silently_drops_chunks() -> None:
    stream = BoundedAudioStream(sample_rate=1000, mode=GenerationMode.ASYNCHRONOUS)
    await stream.put(chunk(0, final=True))
    received = [item async for item in stream]
    assert received == [chunk(0, final=True)]
