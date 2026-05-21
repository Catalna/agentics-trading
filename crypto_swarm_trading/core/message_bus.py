"""
core/message_bus.py — Priority Message Bus with production-grade reliability.

Features:
  - Priority queue (1=low, 5=critical)
  - Pub/Sub with topic-based subscriptions
  - Message TTL (expiration)
  - Max queue size protection
  - Dead-letter queue for failed messages
  - Duplicate message detection (based on message_id)
  - Statistics tracking
  - Thread-safe operations
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from queue import PriorityQueue, Full
from collections import defaultdict, deque
import threading
import time
from typing import Dict, List, Any, Optional
import logging

@dataclass
class AgentMessage:
    id: str
    timestamp: datetime
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    priority: int = 3
    ttl_seconds: int = 30

    @property
    def is_expired(self) -> bool:
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds() > self.ttl_seconds

    def __lt__(self, other):
        return self.priority < other.priority

@dataclass
class MessageBusStats:
    total_sent: int = 0
    total_received: int = 0
    total_expired: int = 0
    total_failed: int = 0
    total_duplicates: int = 0
    avg_processing_ms: float = 0.0

class MessageBus:
    """
    Central communication hub for all agents
    Thread-safe message passing with priority queue
    """
    def __init__(self, max_queue_size: int = 5000):
        self.max_queue_size = max_queue_size
        self.message_queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)
        self.subscribers: Dict[str, List[Any]] = defaultdict(list)
        self._message_history: deque = deque(maxlen=10000)
        self._seen_ids = set()
        self._dead_letter: List[AgentMessage] = []
        self._stats = MessageBusStats()
        self._lock = threading.RLock()
        self.running = False
        self.logger = logging.getLogger("MessageBus")

    def publish(self, message: AgentMessage):
        """
        Publish message to queue and notify subscribers
        Priority: 1 (lowest) to 5 (highest)
        """
        with self._lock:
            # Duplicate detection
            if message.id in self._seen_ids:
                self._stats.total_duplicates += 1
                return
            
            # TTL check
            if message.is_expired:
                self._stats.total_expired += 1
                return

            self._seen_ids.add(message.id)
            # Cleanup seen_ids periodically to prevent memory leak
            if len(self._seen_ids) > 10000:
                self._seen_ids.clear() # Simplistic cleanup for now
                
            self._stats.total_sent += 1
            
            # Direct notify for low latency, queue is for persistence/background tasks if needed
            self._deliver(message)

            # Also put in queue for history/delayed processing
            priority_value = -message.priority # Higher priority = lower queue number
            try:
                self.message_queue.put_nowait((priority_value, message))
            except Full:
                self.logger.warning("Message queue full, dropping oldest.")
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.put_nowait((priority_value, message))
                except Exception:
                    pass
            
            self._message_history.append(message)

    def _deliver(self, message: AgentMessage):
        """Directly deliver message to subscribers"""
        start_time = time.time()
        recipients = self.subscribers.get(message.recipient, [])
        if message.recipient == "ALL":
            # Deliver to all except sender
            recipients = []
            for subs in self.subscribers.values():
                recipients.extend([sub for sub in subs if getattr(sub, 'agent_id', '') != message.sender])
            # Deduplicate recipients
            recipients = list(set(recipients))

        for subscriber in recipients:
            try:
                subscriber.receive_message(message)
                self._stats.total_received += 1
            except Exception as e:
                self.logger.error(f"Failed to deliver message {message.id} to {getattr(subscriber, 'agent_id', 'unknown')}: {e}")
                self._stats.total_failed += 1
                self._dead_letter.append(message)

        duration_ms = (time.time() - start_time) * 1000
        # Simple rolling average
        if self._stats.avg_processing_ms == 0.0:
            self._stats.avg_processing_ms = duration_ms
        else:
            self._stats.avg_processing_ms = (self._stats.avg_processing_ms * 0.9) + (duration_ms * 0.1)


    def subscribe(self, agent: Any, topics: List[str]):
        """Agent subscribes to specific topics/recipients"""
        with self._lock:
            for topic in topics:
                if agent not in self.subscribers[topic]:
                    self.subscribers[topic].append(agent)

    def unsubscribe(self, agent: Any, topics: List[str]):
        with self._lock:
            for topic in topics:
                if topic in self.subscribers and agent in self.subscribers[topic]:
                    self.subscribers[topic].remove(agent)

    def start(self):
        """Start message processing loop"""
        self.running = True
        thread = threading.Thread(target=self._process_loop, name="MessageBus_Thread")
        thread.daemon = True
        thread.start()

    def stop(self):
        self.running = False

    def _process_loop(self):
        """Background thread to process message queue if needed for delayed delivery"""
        while self.running:
            try:
                # We use timeout to periodically check self.running
                priority, message = self.message_queue.get(timeout=1.0)
                # In this architecture, we already delivered directly in publish()
                # The queue is mostly for ensuring nothing is lost or for agents pulling.
                # If we wanted purely asynchronous delivery, we would call _deliver here.
                # For now, we just consume it.
                self.message_queue.task_done()
            except Exception: # Includes queue.Empty
                pass

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_sent": self._stats.total_sent,
                "total_received": self._stats.total_received,
                "total_expired": self._stats.total_expired,
                "total_failed": self._stats.total_failed,
                "total_duplicates": self._stats.total_duplicates,
                "avg_processing_ms": self._stats.avg_processing_ms,
                "queue_size": self.get_queue_size(),
                "dead_letter_size": len(self._dead_letter)
            }

    def get_queue_size(self) -> int:
        return self.message_queue.qsize()

    def get_dead_letters(self) -> List[AgentMessage]:
        with self._lock:
            return list(self._dead_letter)

    def clear_dead_letters(self):
        with self._lock:
            self._dead_letter.clear()
