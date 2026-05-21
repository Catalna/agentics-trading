"""
Agent-specific configurations for retry logic, timeouts, and priorities.
"""

AGENT_CONFIGS = {
    # Layer 1: Data Collection
    'market_data': {'timeout': 10, 'retry': 3, 'delay': 2, 'priority': 5, 'enabled': True, 'cache_ttl': 5},
    'orderbook': {'timeout': 5, 'retry': 2, 'delay': 1, 'priority': 5, 'enabled': True, 'cache_ttl': 3},
    'news': {'timeout': 10, 'retry': 2, 'delay': 2, 'priority': 2, 'enabled': True, 'cache_ttl': 300},
    'social_sentiment': {'timeout': 10, 'retry': 2, 'delay': 2, 'priority': 2, 'enabled': True, 'cache_ttl': 600},
    
    # Layer 2: Analysis
    'technical_analysis': {'timeout': 5, 'retry': 1, 'delay': 0, 'priority': 4, 'enabled': True, 'cache_ttl': 0},
    'ml_prediction': {'timeout': 10, 'retry': 1, 'delay': 0, 'priority': 4, 'enabled': True, 'cache_ttl': 0},
    'sentiment_analysis': {'timeout': 3, 'retry': 1, 'delay': 0, 'priority': 3, 'enabled': True, 'cache_ttl': 0},
    'volatility_regime': {'timeout': 3, 'retry': 1, 'delay': 0, 'priority': 3, 'enabled': True, 'cache_ttl': 0},
    
    # Layer 3: Strategy
    'scalping_strategy': {'timeout': 5, 'retry': 1, 'delay': 0, 'priority': 4, 'enabled': True, 'cache_ttl': 0},
    'swing_strategy': {'timeout': 5, 'retry': 1, 'delay': 0, 'priority': 3, 'enabled': True, 'cache_ttl': 0},
    'mean_reversion': {'timeout': 5, 'retry': 1, 'delay': 0, 'priority': 3, 'enabled': True, 'cache_ttl': 0},
    'breakout': {'timeout': 5, 'retry': 1, 'delay': 0, 'priority': 3, 'enabled': True, 'cache_ttl': 0},
    
    # Layer 4: Risk
    'position_sizer': {'timeout': 3, 'retry': 1, 'delay': 0, 'priority': 5, 'enabled': True, 'cache_ttl': 0},
    'stop_loss': {'timeout': 3, 'retry': 1, 'delay': 0, 'priority': 5, 'enabled': True, 'cache_ttl': 0},
    'portfolio_risk': {'timeout': 5, 'retry': 1, 'delay': 0, 'priority': 4, 'enabled': True, 'cache_ttl': 0},
    'circuit_breaker': {'timeout': 2, 'retry': 1, 'delay': 0, 'priority': 5, 'enabled': True, 'cache_ttl': 0},
    
    # Layer 5: Execution
    'order_router': {'timeout': 10, 'retry': 3, 'delay': 1, 'priority': 5, 'enabled': True, 'cache_ttl': 0},
    'execution_optimizer': {'timeout': 3, 'retry': 1, 'delay': 0, 'priority': 4, 'enabled': True, 'cache_ttl': 0},
    'slippage_monitor': {'timeout': 5, 'retry': 2, 'delay': 1, 'priority': 3, 'enabled': True, 'cache_ttl': 0},
    
    # Layer 6: Meta-Control
    'orchestrator': {'timeout': 10, 'retry': 0, 'delay': 0, 'priority': 5, 'enabled': True, 'cache_ttl': 0},
    'performance_monitor': {'timeout': 5, 'retry': 1, 'delay': 0, 'priority': 2, 'enabled': True, 'cache_ttl': 0},
    'agent_health_monitor': {'timeout': 5, 'retry': 1, 'delay': 0, 'priority': 2, 'enabled': True, 'cache_ttl': 0},
}
