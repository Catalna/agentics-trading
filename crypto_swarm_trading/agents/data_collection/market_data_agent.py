"""
MarketDataAgent — Layer 1 Data Collection
"""

import time
import ccxt
import pandas as pd
from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class MarketDataAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("market_data", config)
        self.symbol = getattr(config, 'SYMBOL', 'BTC/USDT')
        self.timeframes = getattr(config, 'TIMEFRAMES', {'execution': '5m', 'trend': '15m', 'context': '1h', 'major': '4h'})
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        # Note: In a real system, we'd cache with TTL to avoid spamming.
        self._cache = {}
        self._cache_ttl = self.agent_config.get('cache_ttl', 5)

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return True # Triggered by timer/orchestrator typically

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Check cache
        now = time.time()
        if 'last_fetch' in self._cache and (now - self._cache['last_fetch']) < self._cache_ttl:
            return self._cache.get('data')

        data = {}
        for name, tf in self.timeframes.items():
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, tf, limit=100)
            df = self._to_dataframe(ohlcv)
            if not self._validate_ohlcv(df):
                self.logger.warning(f"Invalid OHLCV data for {tf}")
                return None
            data[f"df_{tf}"] = df.to_dict(orient='records') # Simplified for message passing
            
        result = {
            'symbol': self.symbol,
            'data': data,
            'timestamp': now
        }
        
        self._cache['last_fetch'] = now
        self._cache['data'] = result
        
        self.send_message("ALL", "MARKET_DATA", result, priority=5)
        return result

    def _to_dataframe(self, ohlcv) -> pd.DataFrame:
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def _validate_ohlcv(self, df: pd.DataFrame) -> bool:
        if df.empty: return False
        if not (df['high'] >= df['low']).all(): return False
        if not (df['high'] >= df['open']).all(): return False
        if not (df['high'] >= df['close']).all(): return False
        if not (df['volume'] >= 0).all(): return False
        return True
