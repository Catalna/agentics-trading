"""
ExecutionOptimizerAgent — Layer 5 Timing Optimizer
"""

from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class ExecutionOptimizerAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("execution_optimizer", config)

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'orderbook' in input_data and 'technical' in input_data

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        orderbook = input_data['orderbook']
        tech = input_data['technical'].get('raw', {})
        
        spread = orderbook.get('bid_ask_spread_bps', 100)
        vol_ratio = tech.get('vol_ratio', 1.0)
        atr_pct = tech.get('atr', 0) / tech.get('close', 1) if tech.get('close', 1) > 0 else 0
        
        entry_ok = True
        reasons = []
        score = 1.0
        
        if spread > 5:
            entry_ok = False
            reasons.append(f"Spread too high: {spread:.1f} bps")
            score -= 0.3
            
        if vol_ratio < 0.5:
            entry_ok = False
            reasons.append(f"Volume too low: {vol_ratio:.2f}")
            score -= 0.3
            
        if atr_pct > 0.03:
            entry_ok = False
            reasons.append(f"Volatility too high (ATR {atr_pct*100:.1f}%)")
            score -= 0.4
            
        opt_type = "LIMIT" if spread < 5 and vol_ratio > 0.8 else "MARKET"
        
        result = {
            'entry_ok': entry_ok,
            'optimal_order_type': opt_type,
            'timing_score': max(0.0, score),
            'reasons': reasons
        }
        
        self.send_message("ALL", "EXECUTION_OPTIMIZER", result, priority=4)
        return result
