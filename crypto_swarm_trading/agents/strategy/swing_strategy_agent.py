"""
SwingStrategyAgent — Layer 3 Multi-hour Strategy
"""

from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class SwingStrategyAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("swing_strategy", config)
        self.min_confidence = 0.60
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        required = ['technical', 'ml']
        return all(k in input_data for k in required)

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # In a real implementation, we'd look at 15m and 1h data here
        # For this skeleton, we'll use the existing technical data assuming it's multi-tf enriched
        tech = input_data['technical'].get('raw', {})
        ml = input_data['ml']
        
        adx = tech.get('adx', 0)
        
        # Require strong trend
        if adx <= 25:
            return self._hold("Weak trend (ADX <= 25)")
            
        ema5 = tech.get('ema5', 0)
        ema20 = tech.get('ema20', 0)
        rsi = tech.get('rsi', 50)
        
        long_score = 0.0
        short_score = 0.0
        
        if ema5 > ema20:
            if 45 <= rsi <= 75:
                long_score = 0.8
        elif ema5 < ema20:
            if 25 <= rsi <= 55:
                short_score = 0.8
                
        # Blend with ML
        final_long = (ml.get('long_prob', 0.5) * 0.5) + (long_score * 0.5)
        final_short = (ml.get('short_prob', 0.5) * 0.5) + (short_score * 0.5)
        
        confidence = max(final_long, final_short)
        
        if confidence < self.min_confidence:
            return self._hold(f"Confidence {confidence:.3f} below threshold {self.min_confidence}")
            
        signal = "LONG" if final_long > final_short else "SHORT"
        
        result = {
            'signal': signal,
            'confidence': confidence,
            'long_score': final_long,
            'short_score': final_short,
            'reasoning': f"ADX:{adx:.1f} RSI:{rsi:.1f}"
        }
        
        self.send_message("ALL", "STRATEGY_SIGNAL", result, priority=3)
        return result
        
    def _hold(self, reason: str):
        return {
            'signal': "HOLD",
            'confidence': 0.0,
            'long_score': 0.0,
            'short_score': 0.0,
            'reasoning': reason
        }
