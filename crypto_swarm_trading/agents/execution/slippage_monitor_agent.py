"""
SlippageMonitorAgent — Layer 5 Post-trade Analysis
"""

from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class SlippageMonitorAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("slippage_monitor", config)
        self.history = []
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'trade_execution' in input_data

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        exec_data = input_data['trade_execution']
        
        expected = exec_data.get('expected_price', 0)
        actual = exec_data.get('actual_price', 0)
        side = exec_data.get('side', 'LONG')
        
        if expected <= 0 or actual <= 0:
            return None
            
        if side == 'LONG':
            slippage_bps = ((actual - expected) / expected) * 10000
        else:
            slippage_bps = ((expected - actual) / expected) * 10000
            
        self.history.append(slippage_bps)
        if len(self.history) > 100:
            self.history.pop(0)
            
        avg_slippage = sum(self.history) / len(self.history)
        max_slippage = max(self.history)
        
        trend = "STABLE"
        if len(self.history) >= 10:
            recent_avg = sum(self.history[-5:]) / 5
            old_avg = sum(self.history[:5]) / 5
            if recent_avg > old_avg * 1.2:
                trend = "INCREASING"
            elif recent_avg < old_avg * 0.8:
                trend = "DECREASING"
                
        alert = avg_slippage > 10.0 # > 10 bps alert
        
        result = {
            'current_slippage_bps': slippage_bps,
            'avg_slippage_bps': avg_slippage,
            'max_slippage_bps': max_slippage,
            'slippage_trend': trend,
            'alert': alert
        }
        
        if alert:
            self.logger.warning(f"High slippage alert: {avg_slippage:.1f} bps avg")
            
        self.send_message("ALL", "SLIPPAGE_DATA", result, priority=3)
        return result
