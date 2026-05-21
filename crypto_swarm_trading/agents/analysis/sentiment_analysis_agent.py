"""
SentimentAnalysisAgent — Layer 2 Analysis
"""

import time
from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class SentimentAnalysisAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("sentiment_analysis", config)
        self.weights = {
            'fng_score': 0.30,
            'cointelegraph_score': 0.15,
            'decrypt_score': 0.12,
            'bitcoincom_score': 0.12,
            'cryptopotato_score': 0.11,
            'youtube_score': 0.20
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'news' in input_data and 'social' in input_data

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        news_data = input_data.get('news', {})
        social_data = input_data.get('social', {})
        
        now = time.time()
        
        # Merge data and apply weights
        combined_scores = {}
        for k in self.weights.keys():
            if k in news_data: combined_scores[k] = news_data[k]
            elif k in social_data: combined_scores[k] = social_data[k]
            else: combined_scores[k] = 0.5 # Neutral fallback

        # Filter stale
        stale_threshold = 4 * 3600
        if news_data.get('last_updated', 0) < now - stale_threshold:
            self.logger.warning("News data is stale")
        
        composite_score = 0.0
        total_weight = 0.0
        for k, weight in self.weights.items():
            composite_score += combined_scores[k] * weight
            total_weight += weight
            
        if total_weight > 0:
            composite_score /= total_weight
            
        extreme_event = news_data.get('extreme_event', False)
        if extreme_event:
            composite_score -= 0.15
            
        composite_score = max(0.0, min(1.0, composite_score))
        
        # Labeling
        if composite_score >= 0.70: label = "VERY_BULLISH"
        elif composite_score >= 0.55: label = "BULLISH"
        elif composite_score >= 0.45: label = "NEUTRAL"
        elif composite_score > 0.35: label = "BEARISH"
        else: label = "VERY_BEARISH"
        
        # Adjustment
        adjustments = {
            "VERY_BULLISH": +0.05,
            "BULLISH": +0.02,
            "NEUTRAL": 0.0,
            "BEARISH": -0.02,
            "VERY_BEARISH": -0.05
        }
        adj = adjustments[label]
        if extreme_event: adj = -0.15
        
        result = {
            'composite_score': composite_score,
            'label': label,
            'sentiment_adjustment': adj,
            'extreme_event': extreme_event,
            'source_count': len(combined_scores),
            'active_sources': list(combined_scores.keys())
        }
        
        self.send_message("ALL", "SENTIMENT_DATA", result, priority=3)
        return result
