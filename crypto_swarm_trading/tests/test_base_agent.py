"""
Tests for BaseAgent
"""

import pytest
import time
from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent, AgentState

class DummyConfig:
    LOG_LEVEL = 'DEBUG'
    AGENT_ERROR_THRESHOLD = 2
    HEARTBEAT_INTERVAL = 30
    MESSAGE_TTL_SECONDS = 30

class DummyAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("dummy", config)
        self.should_fail = False
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return input_data.get('valid', True)
        
    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.should_fail:
            raise ValueError("Intentional failure")
        return {"processed": True}

def test_agent_initial_state():
    agent = DummyAgent(DummyConfig())
    assert agent.state == AgentState.IDLE
    assert agent.is_healthy is True

def test_successful_execution():
    agent = DummyAgent(DummyConfig())
    result = agent.execute({"valid": True})
    
    assert result == {"processed": True}
    assert agent.state == AgentState.IDLE
    assert agent.performance.success_count == 1
    assert agent.performance.error_count == 0

def test_validation_failure():
    agent = DummyAgent(DummyConfig())
    result = agent.execute({"valid": False})
    
    assert result is None
    # State remains IDLE if validation fails, it's not an internal error
    assert agent.state == AgentState.IDLE 

def test_execution_error_and_recovery():
    agent = DummyAgent(DummyConfig())
    agent.should_fail = True
    
    # Execution should fail gracefully due to safe_agent_execution
    result = agent.execute({"valid": True})
    
    assert result is None
    assert agent.state == AgentState.ERROR
    assert agent._consecutive_errors == 1
    
    # Allow recovery attempt
    agent.should_fail = False
    result2 = agent.execute({"valid": True})
    
    assert result2 == {"processed": True}
    assert agent.state == AgentState.IDLE
    assert agent._consecutive_errors == 0
    assert agent.performance.success_count == 1

def test_max_errors_exceeded():
    agent = DummyAgent(DummyConfig())
    agent.should_fail = True
    
    agent.execute({"valid": True}) # Error 1
    agent.execute({"valid": True}) # Error 2 (Threshold reached)
    
    assert agent.state == AgentState.ERROR
    assert not agent.is_healthy
    
    # 3rd attempt should be skipped entirely
    agent.should_fail = False # Even if fixed
    result = agent.execute({"valid": True})
    assert result is None
    assert agent.state == AgentState.ERROR # Remains stuck in error state
