"""
PortfolioRiskAgent — Layer 4 Account-level Risk
"""

from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class PortfolioRiskAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("portfolio_risk", config)
        self.max_drawdown = getattr(config, 'MAX_DRAWDOWN_PCT', 0.10)

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return True

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Mocking portfolio state
        # In reality, this queries the DB or exchange for current balance, peak balance, open pos
        current_balance = 1000.0
        peak_balance = 1050.0
        open_positions_value = 50.0
        
        drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0
        exposure = open_positions_value / current_balance if current_balance > 0 else 0
        
        # Simplified VaR (Placeholder)
        var_95 = 0.02
        
        risk_level = "SAFE"
        trading_allowed = True
        
        if exposure > 0.10 or drawdown > self.max_drawdown:
            risk_level = "CRITICAL"
            trading_allowed = False
        elif exposure > 0.08:
            risk_level = "DANGER"
        elif exposure > 0.05:
            risk_level = "WARNING"
            
        result = {
            'total_exposure_pct': exposure,
            'var_95': var_95,
            'drawdown_pct': drawdown,
            'risk_level': risk_level,
            'trading_allowed': trading_allowed,
            'max_new_position_size': current_balance * 0.05 if trading_allowed else 0
        }
        
        self.send_message("ALL", "PORTFOLIO_RISK", result, priority=4)
        return result
