"""SSE WebSocket manager for streaming responses."""

import asyncio
import json
from typing import Dict, Callable, Awaitable, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SSEStreamManager:
    """Manager for Server-Sent Events streams."""

    _streams: Dict[str, asyncio.Queue] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def create_stream(self, thread_id: str) -> asyncio.Queue:
        """Create a new stream for a thread."""
        async with self._lock:
            queue = asyncio.Queue()
            self._streams[thread_id] = queue
            return queue

    async def get_stream(self, thread_id: str) -> Optional[asyncio.Queue]:
        """Get existing stream for a thread."""
        return self._streams.get(thread_id)

    async def push_event(self, thread_id: str, event: dict) -> None:
        """Push an event to the stream."""
        queue = self._streams.get(thread_id)
        if queue:
            await queue.put(event)

    async def close_stream(self, thread_id: str) -> None:
        """Close and remove a stream."""
        async with self._lock:
            if thread_id in self._streams:
                del self._streams[thread_id]

    def format_sse(self, event: str, data: dict) -> str:
        """Format data as SSE message."""
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class SSEStream:
    """Context manager for SSE streaming."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def __aiter__(self):
        """Async iterator for SSE events."""
        while True:
            event = await self.queue.get()
            if event is None:  # Sentinel for close
                break
            yield event

    async def push(self, event_type: str, data: dict) -> None:
        """Push an event to the stream."""
        message = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        await self.queue.put(message)

    async def close(self) -> None:
        """Close the stream."""
        await self.queue.put(None)
