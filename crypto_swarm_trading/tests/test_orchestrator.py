"""
Tests for OrchestratorAgent
"""

import pytest
from crypto_swarm_trading.core.orchestrator import OrchestratorAgent

class DummyConfig:
    MAX_QUEUE_SIZE = 100
    DB_PATH = ':memory:'
    LOG_LEVEL = 'CRITICAL'
    AGENT_WEIGHTS = {
        'agentA': 1.0,
        'agentB': 0.5,
        'agentC': 2.0
    }
    AGENT_ERROR_THRESHOLD = 3
    AGENT_COOLDOWN_SECONDS = 60

def test_build_consensus_unanimous():
    orchestrator = OrchestratorAgent(DummyConfig())
    
    outputs = {
        'agentA': {'signal': 'LONG', 'confidence': 0.8},
        'agentB': {'signal': 'LONG', 'confidence': 0.9},
        'agentC': {'signal': 'LONG', 'confidence': 0.7}
    }
    
    result = orchestrator.build_consensus(outputs)
    assert result['signal'] == 'LONG'
    assert result['confidence'] == 1.0 # 100% of votes

def test_build_consensus_weighted():
    orchestrator = OrchestratorAgent(DummyConfig())
    
    # agentA (w=1.0) votes LONG (0.9) -> 0.9
    # agentB (w=0.5) votes LONG (0.8) -> 0.4
    # agentC (w=2.0) votes SHORT (0.9) -> 1.8
    # Total votes = 3.1
    # LONG = 1.3/3.1 = 0.419
    # SHORT = 1.8/3.1 = 0.581
    
    outputs = {
        'agentA': {'signal': 'LONG', 'confidence': 0.9},
        'agentB': {'signal': 'LONG', 'confidence': 0.8},
        'agentC': {'signal': 'SHORT', 'confidence': 0.9}
    }
    
    result = orchestrator.build_consensus(outputs)
    assert result['signal'] == 'SHORT'
    assert result['confidence'] > 0.5
    assert result['confidence'] < 0.6
    
def test_build_consensus_all_hold():
    orchestrator = OrchestratorAgent(DummyConfig())
    
    outputs = {
        'agentA': {'signal': 'HOLD', 'confidence': 0.0},
        'agentB': {'signal': 'HOLD', 'confidence': 0.0}
    }
    
    result = orchestrator.build_consensus(outputs)
    assert result['signal'] == 'HOLD'
    assert result['confidence'] == 0.0

def test_disable_agent():
    orchestrator = OrchestratorAgent(DummyConfig())
    orchestrator._disable_agent('agentA')
    
    status = orchestrator.get_system_status()
    assert 'agentA' in status['disabled_agents']
    
    active = orchestrator._get_active_agents(['agentA', 'agentB'])
    assert active == ['agentB']
