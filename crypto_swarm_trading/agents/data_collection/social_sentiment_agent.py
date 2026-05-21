"""
SocialSentimentAgent — Layer 1 Data Collection
"""

import time
import feedparser
from typing import Dict, Any, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from crypto_swarm_trading.core.base_agent import BaseAgent

class SocialSentimentAgent(BaseAgent):
    BULLISH_KEYWORDS = ['bullish', 'breakout', 'surge', 'rally', 'adoption', 'ath', 'accumulation', 'recovery']
    BEARISH_KEYWORDS = ['hack', 'liquidation', 'lawsuit', 'ban', 'dump', 'crash', 'regulation', 'fud', 'selloff']

    def __init__(self, config):
        super().__init__("social_sentiment", config)
        # Using YouTube RSS for channels (dummy channel IDs for example)
        self.channels = {
            'coin_bureau': 'https://www.youtube.com/feeds/videos.xml?channel_id=UCqK_GSMbpiV8spgD3ZGloSw',
            'altcoin_daily': 'https://www.youtube.com/feeds/videos.xml?channel_id=UCbLhGKVY-bJPcaALjdIGbrQ'
        }
        self.analyzer = SentimentIntensityAnalyzer()
        self._cache = {}
        self._cache_ttl = 600

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return True

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        now = time.time()
        
        if 'last_fetch' in self._cache and (now - self._cache['last_fetch']) < self._cache_ttl:
            return self._cache['data']

        total_score = 0
        items_analyzed = 0

        for name, url in self.channels.items():
            try:
                feed = feedparser.parse(url)
                if not feed.entries: continue
                
                for entry in feed.entries[:5]:
                    text = entry.title.lower()
                    
                    vs = self.analyzer.polarity_scores(text)['compound']
                    tb = TextBlob(text).sentiment.polarity
                    
                    bull_count = sum(1 for k in self.BULLISH_KEYWORDS if k in text)
                    bear_count = sum(1 for k in self.BEARISH_KEYWORDS if k in text)
                    
                    base = (vs + tb) / 2
                    base += (bull_count * 0.1) - (bear_count * 0.1)
                    
                    total_score += max(-1.0, min(1.0, base))
                    items_analyzed += 1
            except Exception as e:
                self.logger.warning(f"Failed to fetch social sentiment from {name}: {e}")
                
        if items_analyzed > 0:
            avg_score = total_score / items_analyzed
            norm_score = (avg_score + 1) / 2
            
            label = "NEUTRAL"
            if norm_score >= 0.65: label = "BULLISH"
            elif norm_score <= 0.35: label = "BEARISH"
            
            result = {
                'youtube_score': norm_score,
                'sentiment_label': label,
                'items_analyzed': items_analyzed,
                'last_updated': now
            }
        else:
            result = {
                'youtube_score': 0.5,
                'sentiment_label': "NEUTRAL",
                'items_analyzed': 0,
                'last_updated': now
            }

        self._cache['last_fetch'] = now
        self._cache['data'] = result
        self.send_message("ALL", "SOCIAL_DATA", result, priority=2)
        
        return result
