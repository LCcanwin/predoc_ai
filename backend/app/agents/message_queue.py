"""Per-session user message queue."""

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Optional


@dataclass
class QueuedMessage:
    """Message waiting to be processed by the agent pipeline."""

    thread_id: str
    user_id: str
    content: str
    created_at: str


class UserMessageQueue:
    """Small in-memory queue that decouples API input from agent processing."""

    def __init__(self) -> None:
        self._queues: dict[str, deque[QueuedMessage]] = defaultdict(deque)
        self._lock = Lock()

    def enqueue(self, thread_id: str, user_id: str, content: str) -> QueuedMessage:
        item = QueuedMessage(
            thread_id=thread_id,
            user_id=user_id,
            content=content,
            created_at=datetime.utcnow().isoformat(),
        )
        with self._lock:
            self._queues[thread_id].append(item)
        return item

    def dequeue(self, thread_id: str) -> Optional[QueuedMessage]:
        with self._lock:
            queue = self._queues.get(thread_id)
            if not queue:
                return None
            return queue.popleft()

    def depth(self, thread_id: str) -> int:
        with self._lock:
            return len(self._queues.get(thread_id, ()))
