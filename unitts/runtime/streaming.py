from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from unitts.contracts import AudioChunk, GenerationMode, StreamBackpressureError


class BoundedAudioStream:
    """Ordered audio queue with explicit slow-client behavior and cancellation."""

    def __init__(self, *, sample_rate: int, max_buffered_seconds: float = 2.0, slow_client_timeout: float = 5.0, mode: GenerationMode = GenerationMode.REALTIME) -> None:
        self.sample_rate = sample_rate
        self.max_samples = max(1, int(sample_rate * max_buffered_seconds))
        self.slow_client_timeout = slow_client_timeout
        self.mode = mode
        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        self._buffered_samples = 0
        self._next_sequence = 0
        self._cancelled = asyncio.Event()
        self._capacity_changed = asyncio.Event()
        self._capacity_changed.set()

    async def put(self, chunk: AudioChunk) -> None:
        if self._cancelled.is_set():
            raise asyncio.CancelledError
        if chunk.sequence != self._next_sequence:
            raise ValueError(f"Expected chunk sequence {self._next_sequence}, got {chunk.sequence}")
        chunk_samples = len(chunk.data) // 2 if chunk.encoding == "pcm_s16le" else 0
        deadline = None if self.mode is GenerationMode.ASYNCHRONOUS else self.slow_client_timeout

        async def wait_for_capacity() -> None:
            while self._buffered_samples + chunk_samples > self.max_samples:
                self._capacity_changed.clear()
                await self._capacity_changed.wait()
        try:
            if deadline is None:
                await wait_for_capacity()
            else:
                await asyncio.wait_for(wait_for_capacity(), timeout=deadline)
        except TimeoutError as exc:
            self.cancel()
            raise StreamBackpressureError("Realtime client remained too slow") from exc
        await self._queue.put(chunk)
        self._buffered_samples += chunk_samples
        self._next_sequence += 1

    async def __aiter__(self) -> AsyncIterator[AudioChunk]:
        while not self._cancelled.is_set():
            chunk = await self._queue.get()
            self._buffered_samples = max(0, self._buffered_samples - (len(chunk.data) // 2 if chunk.encoding == "pcm_s16le" else 0))
            self._capacity_changed.set()
            yield chunk
            if chunk.final:
                return

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()
