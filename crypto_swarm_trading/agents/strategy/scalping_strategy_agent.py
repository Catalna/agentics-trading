"""
ScalpingStrategyAgent — Layer 3 Primary Strategy
"""

from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class ScalpingStrategyAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("scalping_strategy", config)
        self.min_confidence = getattr(config, 'MIN_SIGNAL_CONFIDENCE', 0.55)
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        required = ['technical', 'ml', 'sentiment', 'volatility']
        return all(k in input_data for k in required)

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tech = input_data['technical']
        ml = input_data['ml']
        sent = input_data['sentiment']
        vol = input_data['volatility']
        
        if sent.get('extreme_event', False):
            return self._hold("Extreme sentiment event detected")
            
        long_score = 0.0
        short_score = 0.0
        
        # 1. Technical Scoring (20% weight)
        tech_norm = tech.get('normalized', {})
        tech_raw = tech.get('raw', {})
        
        # LONG Tech
        if tech_norm.get('ema5_dist', 0) > 0 and tech_norm.get('ema20_dist', 0) > 0:
            if tech_raw.get('ema5', 0) > tech_raw.get('ema20', 0):
                long_score += 1.0
            else:
                long_score += 0.4
                
        rsi = tech_raw.get('rsi', 50)
        if 50 <= rsi <= 70: long_score += 1.0
        elif 45 <= rsi < 50: long_score += 0.4
        elif rsi > 70: long_score -= 0.5
        
        if tech_norm.get('macd_norm', 0) > 0 and tech_norm.get('macd_slope', 0) > 0: long_score += 1.0
        elif tech_norm.get('macd_norm', 0) > 0: long_score += 0.6
        elif tech_norm.get('macd_slope', 0) > 0: long_score += 0.3
        
        adx = tech_raw.get('adx', 0)
        if adx > 20: long_score += 1.0
        elif adx > 15: long_score += 0.5
        
        vol_ratio = tech_raw.get('vol_ratio', 1.0)
        if vol_ratio > 1.2: long_score += 1.0
        elif vol_ratio > 1.0: long_score += 0.5
        
        # SHORT Tech
        if tech_norm.get('ema5_dist', 0) < 0 and tech_norm.get('ema20_dist', 0) < 0:
            if tech_raw.get('ema5', 0) < tech_raw.get('ema20', 0):
                short_score += 1.0
            else:
                short_score += 0.4
                
        if 30 <= rsi <= 50: short_score += 1.0
        elif 50 < rsi <= 55: short_score += 0.4
        elif rsi < 30: short_score -= 0.5
        
        if tech_norm.get('macd_norm', 0) < 0 and tech_norm.get('macd_slope', 0) < 0: short_score += 1.0
        elif tech_norm.get('macd_norm', 0) < 0: short_score += 0.6
        elif tech_norm.get('macd_slope', 0) < 0: short_score += 0.3
        
        if adx > 20: short_score += 1.0
        elif adx > 15: short_score += 0.5
        
        if vol_ratio > 1.2: short_score += 1.0
        elif vol_ratio > 1.0: short_score += 0.5
        
        # Normalize tech scores (max possible is roughly 5.0)
        tech_long = max(0, min(1.0, long_score / 5.0))
        tech_short = max(0, min(1.0, short_score / 5.0))
        
        # 2. ML Scoring (70% weight)
        ml_long = ml.get('long_prob', 0.5)
        ml_short = ml.get('short_prob', 0.5)
        
        # 3. Sentiment Scoring (10% weight)
        sent_adj = sent.get('sentiment_adjustment', 0.0)
        
        # Composite
        final_long = (ml_long * 0.70) + (tech_long * 0.20) + (max(0, sent_adj) * 0.10)
        final_short = (ml_short * 0.70) + (tech_short * 0.20) + (max(0, -sent_adj) * 0.10)
        
        confidence = max(final_long, final_short)
        
        if confidence < self.min_confidence:
            return self._hold(f"Confidence {confidence:.3f} below threshold {self.min_confidence}")
            
        signal = "LONG" if final_long > final_short else "SHORT"
        
        result = {
            'signal': signal,
            'confidence': confidence,
            'long_score': final_long,
            'short_score': final_short,
            'reasoning': f"Tech L:{tech_long:.2f} S:{tech_short:.2f} | ML L:{ml_long:.2f} S:{ml_short:.2f}"
        }
        
        self.send_message("ALL", "STRATEGY_SIGNAL", result, priority=4)
        return result
        
    def _hold(self, reason: str):
        return {
            'signal': "HOLD",
            'confidence': 0.0,
            'long_score': 0.0,
            'short_score': 0.0,
            'reasoning': reason
        }
