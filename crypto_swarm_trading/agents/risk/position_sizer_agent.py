"""
PositionSizerAgent — Layer 4 Kelly Criterion Sizing
"""

from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent
import logging

class PositionSizerAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("position_sizer", config)
        self.kelly_fraction = getattr(config, 'KELLY_FRACTION', 0.25)
        self.max_pos_pct = getattr(config, 'MAX_POSITION_SIZE_PCT', 0.05)
        self.margin_alloc = 0.05 # Fallback
        
        # In a real impl, we'd query DB for this
        self._mock_win_rate = 0.55 
        self._mock_avg_win = 100.0
        self._mock_avg_loss = 80.0

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'signal' in input_data and 'confidence' in input_data

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        signal = input_data.get('signal', 'HOLD')
        if signal == 'HOLD':
            return None
            
        confidence = input_data.get('confidence', 0.5)
        
        # Mock equity for now
        equity = 1000.0
        
        # Kelly calculation
        b = self._mock_avg_win / self._mock_avg_loss if self._mock_avg_loss > 0 else 1.0
        p = self._mock_win_rate
        q = 1.0 - p
        
        kelly = (p * b - q) / b if b > 0 else 0
        kelly_pct = max(0, kelly * self.kelly_fraction)
        
        # Adjustments
        confidence_mult = min(1.2, confidence * 1.5)
        vol_mult = input_data.get('volatility', {}).get('position_multiplier', 1.0)
        
        raw_size = equity * kelly_pct * confidence_mult * vol_mult
        
        # Caps
        max_allowed = equity * self.max_pos_pct
        if raw_size <= 0:
            raw_size = equity * self.margin_alloc # Fallback
            
        final_size_usd = min(raw_size, max_allowed)
        
        if final_size_usd < 10.0: # min notional
            final_size_usd = 10.0
            
        # Simplified quantity (need actual price, assuming price in technical data)
        price = input_data.get('technical', {}).get('raw', {}).get('close', 50000.0)
        leverage = input_data.get('volatility', {}).get('suggested_leverage', 20)
        
        quantity = (final_size_usd * leverage) / price
        
        result = {
            'position_size_usd': final_size_usd,
            'quantity': quantity,
            'kelly_pct': kelly_pct,
            'leverage_used': leverage,
            'reasoning': f"Kelly {kelly_pct:.3f} * VolMult {vol_mult:.2f} * ConfMult {confidence_mult:.2f}"
        }
        
        self.send_message("ALL", "POSITION_SIZE_DATA", result, priority=5)
        return result
