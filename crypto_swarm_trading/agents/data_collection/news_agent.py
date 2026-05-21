"""
NewsAgent — Layer 1 Data Collection
"""

import time
import feedparser
import requests
from typing import Dict, Any, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from crypto_swarm_trading.core.base_agent import BaseAgent

class NewsAgent(BaseAgent):
    BULLISH_KEYWORDS = ['bullish', 'breakout', 'surge', 'rally', 'adoption', 'ath', 'accumulation', 'recovery']
    BEARISH_KEYWORDS = ['hack', 'liquidation', 'lawsuit', 'ban', 'dump', 'crash', 'regulation', 'fud', 'selloff']
    EXTREME_KEYWORDS = ['exchange hack', 'etf rejected', 'mass liquidation', 'flash crash', 'interest rate', 'ban crypto']

    def __init__(self, config):
        super().__init__("news", config)
        self.sources = {
            'cointelegraph': 'https://cointelegraph.com/rss',
            'bitcoincom': 'https://news.bitcoin.com/feed/'
            # Decrypt, CryptoPotato omitted for brevity in this mock implementation, can be added
        }
        self.analyzer = SentimentIntensityAnalyzer()
        self._cache = {}
        self._cache_ttl_rss = 300
        self._cache_ttl_fng = 3600

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return True

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        now = time.time()
        
        # Check cache
        if 'last_fetch' in self._cache and (now - self._cache['last_fetch']) < min(self._cache_ttl_rss, self._cache_ttl_fng):
            return self._cache['data']

        result = {'last_updated': now, 'extreme_event': False}

        # 1. Fetch FnG
        try:
            fng_data = requests.get('https://api.alternative.me/fng/?limit=1', timeout=10).json()
            score = float(fng_data['data'][0]['value']) / 100.0
            result['fng_score'] = score
            result['fng_label'] = fng_data['data'][0]['value_classification']
        except Exception as e:
            self.logger.warning(f"Failed to fetch FnG: {e}")
            result['fng_score'] = 0.5
            result['fng_label'] = "Neutral"

        # 2. Fetch RSS
        for name, url in self.sources.items():
            try:
                feed = feedparser.parse(url)
                if not feed.entries: continue
                
                total_score = 0
                count = 0
                for entry in feed.entries[:5]:
                    text = (entry.title + " " + getattr(entry, 'summary', '')).lower()
                    
                    # Extreme check
                    if any(ek in text for ek in self.EXTREME_KEYWORDS):
                        result['extreme_event'] = True
                        
                    # NLP
                    vs = self.analyzer.polarity_scores(text)['compound']
                    tb = TextBlob(text).sentiment.polarity
                    
                    # Keyword boost
                    bull_count = sum(1 for k in self.BULLISH_KEYWORDS if k in text)
                    bear_count = sum(1 for k in self.BEARISH_KEYWORDS if k in text)
                    
                    base = (vs + tb) / 2
                    base += (bull_count * 0.1) - (bear_count * 0.1)
                    
                    total_score += max(-1.0, min(1.0, base))
                    count += 1
                    
                if count > 0:
                    # Normalize to 0-1
                    avg_score = (total_score / count)
                    norm_score = (avg_score + 1) / 2
                    result[f'{name}_score'] = norm_score
                else:
                    result[f'{name}_score'] = 0.5

            except Exception as e:
                self.logger.warning(f"Failed to fetch {name}: {e}")
                result[f'{name}_score'] = 0.5
                
        self._cache['last_fetch'] = now
        self._cache['data'] = result
        self.send_message("ALL", "NEWS_DATA", result, priority=2)
        
        return result
