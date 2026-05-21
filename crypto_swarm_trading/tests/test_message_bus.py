"""
Tests for MessageBus
"""

import pytest
import time
from datetime import datetime, timezone, timedelta
from crypto_swarm_trading.core.message_bus import MessageBus, AgentMessage

class MockSubscriber:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.received = []
        self.should_fail = False
        
    def receive_message(self, message):
        if self.should_fail:
            raise Exception("Mock failure")
        self.received.append(message)

def test_publish_subscribe_direct_delivery():
    bus = MessageBus()
    sub1 = MockSubscriber("sub1")
    
    bus.subscribe(sub1, ["TOPIC_A"])
    
    msg = AgentMessage(
        id="1", timestamp=datetime.now(timezone.utc),
        sender="sender1", recipient="TOPIC_A",
        message_type="DATA", payload={"val": 1}
    )
    
    bus.publish(msg)
    
    assert len(sub1.received) == 1
    assert sub1.received[0].id == "1"
    
    stats = bus.get_stats()
    assert stats['total_sent'] == 1
    assert stats['total_received'] == 1

def test_duplicate_detection():
    bus = MessageBus()
    sub1 = MockSubscriber("sub1")
    bus.subscribe(sub1, ["TOPIC_A"])
    
    msg = AgentMessage(
        id="1", timestamp=datetime.now(timezone.utc),
        sender="sender1", recipient="TOPIC_A",
        message_type="DATA", payload={"val": 1}
    )
    
    bus.publish(msg)
    bus.publish(msg) # Duplicate
    
    assert len(sub1.received) == 1 # Only received once
    
    stats = bus.get_stats()
    assert stats['total_duplicates'] == 1

def test_ttl_expiration():
    bus = MessageBus()
    sub1 = MockSubscriber("sub1")
    bus.subscribe(sub1, ["TOPIC_A"])
    
    # Create expired message (1 minute ago with 30s TTL)
    old_time = datetime.now(timezone.utc) - timedelta(seconds=60)
    msg = AgentMessage(
        id="1", timestamp=old_time,
        sender="sender1", recipient="TOPIC_A",
        message_type="DATA", payload={}, ttl_seconds=30
    )
    
    bus.publish(msg)
    
    assert len(sub1.received) == 0
    stats = bus.get_stats()
    assert stats['total_expired'] == 1

def test_dead_letter_queue():
    bus = MessageBus()
    sub1 = MockSubscriber("sub1")
    sub1.should_fail = True
    
    bus.subscribe(sub1, ["TOPIC_A"])
    
    msg = AgentMessage(
        id="1", timestamp=datetime.now(timezone.utc),
        sender="sender1", recipient="TOPIC_A",
        message_type="DATA", payload={}
    )
    
    bus.publish(msg)
    
    assert len(sub1.received) == 0
    stats = bus.get_stats()
    assert stats['total_failed'] == 1
    assert stats['dead_letter_size'] == 1
    
    dlq = bus.get_dead_letters()
    assert len(dlq) == 1
    assert dlq[0].id == "1"
