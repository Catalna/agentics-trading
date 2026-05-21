"""
CircuitBreakerAgent — Layer 4 Safety Mechanisms
"""

from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class CircuitBreakerAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("circuit_breaker", config)
        self.max_daily_loss = getattr(config, 'MAX_DAILY_LOSS_PCT', 0.02)
        self.manual_halt = False
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return True

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Mocking daily PnL and other checks
        daily_pnl = -0.005 # -0.5%
        drawdown_pct = input_data.get('portfolio', {}).get('drawdown_pct', 0)
        extreme_event = input_data.get('sentiment', {}).get('extreme_event', False)
        
        state = "CLEAR"
        reason = ""
        trading_allowed = True
        
        if self.manual_halt:
            state = "MANUAL_HALT"
            reason = "Manual halt engaged"
            trading_allowed = False
        elif daily_pnl <= -self.max_daily_loss:
            state = "DAILY_LOSS_LIMIT"
            reason = f"Daily loss exceeded: {daily_pnl:.3f}"
            trading_allowed = False
        elif drawdown_pct >= getattr(self.config, 'MAX_DRAWDOWN_PCT', 0.10):
            state = "DRAWDOWN_LIMIT"
            reason = f"Max drawdown exceeded: {drawdown_pct:.3f}"
            trading_allowed = False
        elif extreme_event:
            state = "EXTREME_EVENT"
            reason = "Extreme sentiment event detected"
            trading_allowed = False
            
        result = {
            'trading_allowed': trading_allowed,
            'reason': reason,
            'circuit_state': state,
            'daily_pnl': daily_pnl,
            'drawdown_pct': drawdown_pct
        }
        
        self.send_message("ALL", "CIRCUIT_BREAKER", result, priority=5)
        return result
