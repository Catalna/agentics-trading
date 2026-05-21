"""
MeanReversionAgent — Layer 3 Ranging Market Strategy
"""

from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class MeanReversionAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("mean_reversion", config)
        self.min_confidence = 0.60
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'technical' in input_data and 'volatility' in input_data

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tech = input_data['technical'].get('raw', {})
        regime = input_data['technical'].get('regime', 'RANGING')
        
        if regime == "TRENDING" or tech.get('adx', 100) >= 20:
            return self._hold("Market is TRENDING, MeanReversion disabled", False)
            
        rsi = tech.get('rsi', 50)
        close = tech.get('close', 0)
        bb_lower = tech.get('bb_lower', 0)
        bb_upper = tech.get('bb_upper', 0)
        vol_ratio = tech.get('vol_ratio', 0)
        
        long_score = 0.0
        short_score = 0.0
        
        if rsi < 35 and close < bb_lower and vol_ratio > 1.0:
            long_score = 0.8
            
        if rsi > 65 and close > bb_upper and vol_ratio > 1.0:
            short_score = 0.8
            
        confidence = max(long_score, short_score)
        
        if confidence < self.min_confidence:
            return self._hold("Conditions not met", True)
            
        signal = "LONG" if long_score > short_score else "SHORT"
        
        result = {
            'signal': signal,
            'confidence': confidence,
            'long_score': long_score,
            'short_score': short_score,
            'regime_valid': True,
            'reasoning': f"RSI:{rsi:.1f} BB bounds reached"
        }
        
        self.send_message("ALL", "STRATEGY_SIGNAL", result, priority=3)
        return result
        
    def _hold(self, reason: str, regime_valid: bool):
        return {
            'signal': "HOLD",
            'confidence': 0.0,
            'long_score': 0.0,
            'short_score': 0.0,
            'regime_valid': regime_valid,
            'reasoning': reason
        }
