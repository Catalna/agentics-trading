"""
BreakoutAgent — Layer 3 Momentum Strategy
"""

from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class BreakoutAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("breakout", config)
        self.min_confidence = 0.65 # Stricter
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'technical' in input_data

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tech = input_data['technical'].get('raw', {})
        
        adx = tech.get('adx', 0)
        if adx <= 15:
            return self._hold("Momentum too low (ADX <= 15)")
            
        close = tech.get('close', 0)
        bb_lower = tech.get('bb_lower', 0)
        bb_upper = tech.get('bb_upper', 0)
        vol_ratio = tech.get('vol_ratio', 0)
        macd_hist = tech.get('macd_hist', 0)
        
        long_score = 0.0
        short_score = 0.0
        breakout_dir = None
        
        if close > bb_upper and vol_ratio > 2.0 and macd_hist > 0:
            long_score = 0.8
            breakout_dir = "UP"
            
        if close < bb_lower and vol_ratio > 2.0 and macd_hist < 0:
            short_score = 0.8
            breakout_dir = "DOWN"
            
        confidence = max(long_score, short_score)
        
        if confidence < self.min_confidence:
            return self._hold("Breakout conditions not met")
            
        signal = "LONG" if long_score > short_score else "SHORT"
        
        result = {
            'signal': signal,
            'confidence': confidence,
            'long_score': long_score,
            'short_score': short_score,
            'breakout_direction': breakout_dir,
            'volume_confirmation': vol_ratio > 2.0,
            'reasoning': f"VolRatio:{vol_ratio:.2f} MACD:{macd_hist:.2f}"
        }
        
        self.send_message("ALL", "STRATEGY_SIGNAL", result, priority=3)
        return result
        
    def _hold(self, reason: str):
        return {
            'signal': "HOLD",
            'confidence': 0.0,
            'long_score': 0.0,
            'short_score': 0.0,
            'breakout_direction': None,
            'volume_confirmation': False,
            'reasoning': reason
        }
