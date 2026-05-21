"""
OrderbookAgent — Layer 1 Data Collection
"""

import time
import ccxt
from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class OrderbookAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("orderbook", config)
        self.symbol = getattr(config, 'SYMBOL', 'BTC/USDT')
        self.depth = getattr(config, 'ORDERBOOK_DEPTH', 20)
        self.whale_threshold = getattr(config, 'WHALE_THRESHOLD_BTC', 5.0)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self._cache = {}
        self._cache_ttl = self.agent_config.get('cache_ttl', 3)

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return True

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        now = time.time()
        if 'last_fetch' in self._cache and (now - self._cache['last_fetch']) < self._cache_ttl:
            return self._cache.get('data')

        try:
            ob = self._fetch_orderbook()
            if not ob['bids'] or not ob['asks']:
                raise ValueError("Empty orderbook")

            spread_bps = self._calculate_spread(ob)
            imbalance = self._calculate_imbalance(ob)
            liq_score = self._calculate_liquidity_score(ob)
            whale_walls = self._detect_whale_walls(ob)
            slippage = self._estimate_slippage(ob)

            liq_sufficient = liq_score > 0.6 and spread_bps < 10 and abs(imbalance) < 0.7

            result = {
                'bid_ask_spread_bps': spread_bps,
                'orderbook_imbalance': imbalance,
                'liquidity_score': liq_score,
                'whale_walls': whale_walls,
                'estimated_slippage': slippage,
                'liquidity_sufficient': liq_sufficient,
                'timestamp': now
            }
            
            self._cache['last_fetch'] = now
            self._cache['data'] = result
            self.send_message("ALL", "ORDERBOOK_DATA", result, priority=5)
            return result

        except Exception as e:
            self.logger.error(f"Orderbook fetch failed: {e}")
            degraded = {'liquidity_sufficient': False, 'error': str(e), 'timestamp': now}
            self.send_message("ALL", "ORDERBOOK_DATA", degraded, priority=5)
            return degraded

    def _fetch_orderbook(self):
        return self.exchange.fetch_order_book(self.symbol, limit=self.depth)

    def _calculate_spread(self, ob):
        bid = ob['bids'][0][0]
        ask = ob['asks'][0][0]
        return ((ask - bid) / bid) * 10000

    def _calculate_imbalance(self, ob):
        bid_vol = sum(v for p, v in ob['bids'][:5])
        ask_vol = sum(v for p, v in ob['asks'][:5])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0

    def _calculate_liquidity_score(self, ob):
        # Simplified liquidity score
        bid_vol = sum(v for p, v in ob['bids'][:10])
        ask_vol = sum(v for p, v in ob['asks'][:10])
        total_vol = bid_vol + ask_vol
        vol_score = min(total_vol / 100.0, 1.0)
        
        spread = self._calculate_spread(ob)
        spread_score = max(0.0, 1.0 - (spread / 20.0))
        
        return 0.6 * vol_score + 0.4 * spread_score

    def _detect_whale_walls(self, ob):
        walls = []
        for p, v in ob['bids']:
            if v > self.whale_threshold: walls.append({'side': 'BID', 'price': p, 'vol': v})
        for p, v in ob['asks']:
            if v > self.whale_threshold: walls.append({'side': 'ASK', 'price': p, 'vol': v})
        return walls

    def _estimate_slippage(self, ob):
        trade_size = 0.1
        filled = 0
        cost = 0
        for p, v in ob['asks']:
            take = min(v, trade_size - filled)
            cost += take * p
            filled += take
            if filled >= trade_size: break
        
        if filled == 0: return 100.0
        avg_price = cost / filled
        best_ask = ob['asks'][0][0]
        return ((avg_price - best_ask) / best_ask) * 100
