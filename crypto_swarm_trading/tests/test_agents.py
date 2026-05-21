"""
Tests for Key Agents
"""

import pytest
import pandas as pd
from agents.data_collection.orderbook_agent import OrderbookAgent
from agents.risk.position_sizer_agent import PositionSizerAgent
from agents.risk.circuit_breaker_agent import CircuitBreakerAgent

class DummyConfig:
    SYMBOL = 'BTC/USDT'
    ORDERBOOK_DEPTH = 20
    WHALE_THRESHOLD_BTC = 5.0
    KELLY_FRACTION = 0.25
    MAX_POSITION_SIZE_PCT = 0.05
    MAX_DAILY_LOSS_PCT = 0.02
    MAX_DRAWDOWN_PCT = 0.10

def test_orderbook_imbalance():
    agent = OrderbookAgent(DummyConfig())
    
    mock_ob = {
        'bids': [[100, 10], [99, 10], [98, 10], [97, 10], [96, 10]], # Vol = 50
        'asks': [[101, 5], [102, 5], [103, 5], [104, 5], [105, 5]]   # Vol = 25
    }
    
    imb = agent._calculate_imbalance(mock_ob)
    # (50 - 25) / 75 = 25 / 75 = 0.333...
    assert round(imb, 3) == 0.333
    
def test_position_sizer_kelly():
    agent = PositionSizerAgent(DummyConfig())
    # Mock data inside agent init: win_rate=0.55, avg_win=100, avg_loss=80
    # b = 1.25. Kelly = (0.55 * 1.25 - 0.45) / 1.25 = (0.6875 - 0.45) / 1.25 = 0.2375 / 1.25 = 0.19
    # Kelly Pct = 0.19 * 0.25 = 0.0475 (4.75%)
    
    input_data = {
        'signal': 'LONG',
        'confidence': 0.8,
        'volatility': {'position_multiplier': 1.0, 'suggested_leverage': 10},
        'technical': {'raw': {'close': 50000.0}}
    }
    
    result = agent.process(input_data)
    assert result is not None
    assert round(result['kelly_pct'], 4) == 0.0475
    
    # Raw size = 1000 * 0.0475 * (0.8*1.5=1.2) * 1.0 = 57.0 USD
    # Max allowed = 1000 * 0.05 = 50.0 USD
    assert result['position_size_usd'] == 50.0

def test_circuit_breaker_loss():
    agent = CircuitBreakerAgent(DummyConfig())
    
    # Process internal mock has daily_pnl = -0.005
    # Max daily loss is 0.02, so it should be CLEAR
    result = agent.process({})
    assert result['circuit_state'] == 'CLEAR'
    assert result['trading_allowed'] is True
    
    # But if we inject an extreme event
    input_data = {'sentiment': {'extreme_event': True}}
    result2 = agent.process(input_data)
    assert result2['circuit_state'] == 'EXTREME_EVENT'
    assert result2['trading_allowed'] is False
