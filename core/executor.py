import logging
import time
from typing import Dict, Any, Optional

import config

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

logger = logging.getLogger(__name__)

class TradeExecutor:
    def __init__(self):
        self.symbol    = config.SYMBOL
        self.leverage  = config.DEFAULT_LEVERAGE
        self.trade_history = []

        if not CCXT_AVAILABLE:
            raise RuntimeError("ccxt library not found. Install it with: pip install ccxt")

        # ccxt.binanceusdm = dedicated USDT-M Futures
        self.exchange = ccxt.binanceusdm({
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_SECRET_KEY,
            'enableRateLimit': True,
        })

        if config.USE_TESTNET:
            # Bypass CCXT's deprecated sandbox for binanceusdm testnet via manual URL override
            self.exchange.urls['api']['fapiPublic']   = 'https://testnet.binancefuture.com/fapi/v1'
            self.exchange.urls['api']['fapiPrivate']  = 'https://testnet.binancefuture.com/fapi/v1'
            self.exchange.urls['api']['fapiPublicV2'] = 'https://testnet.binancefuture.com/fapi/v2'
            self.exchange.urls['api']['fapiPrivateV2']= 'https://testnet.binancefuture.com/fapi/v2'
            logger.info("Connected to Binance Futures Testnet (Raw API mode — CCXT sandbox bypassed).")
        else:
            try:
                self.exchange.load_markets()
                logger.info(f"Connected to {self.exchange.id}. Markets loaded.")
                try:
                    self.exchange.set_leverage(self.leverage, self.symbol)
                    logger.info(f"Leverage set to {self.leverage}x for {self.symbol}")
                except Exception as e:
                    logger.warning(f"Failed to set leverage: {e}")
            except Exception as e:
                logger.error(f"Failed to connect to Binance: {e}")
                raise ValueError("Gagal konek Binance API! Pastikan API Key Anda valid.")

    # ──────────────────────────────────────────────────────────────────────────
    def execute_trade(self, decision: Dict[str, Any], current_price: float) -> bool:
        """Executes a trade based on the Chief Supervisor's decision."""
        if not decision or not isinstance(decision, dict):
            logger.warning("Invalid decision received (None or non-dict). Skipping.")
            return False

        action     = str(decision.get("final_decision") or "WAIT").upper()
        confidence = float(decision.get("confidence") or 0.0)

        if action == "WAIT" or confidence < config.MIN_CONFIDENCE:
            logger.info(f"[SKIP] Action: {action} | Confidence: {confidence:.2f} (min: {config.MIN_CONFIDENCE})")
            return False

        sl_pct = float(decision.get("sl_pct") or config.DEFAULT_SL_PCT)
        tp_pct = float(decision.get("tp_pct") or config.DEFAULT_TP_PCT)

        # Safety cap for 100x: SL must not exceed 0.8% (liquidation at ~1%)
        sl_pct = min(sl_pct, 0.008)
        tp_pct = min(tp_pct, 0.016)

        logger.info(f"[EXECUTE] {action} {self.symbol} @ ~${current_price:,.2f} (Conf: {confidence:.2f})")

        if config.USE_TESTNET:
            return self._testnet_execute(action, current_price, sl_pct, tp_pct)
        else:
            return self._mainnet_execute(action, current_price, sl_pct, tp_pct)

    # ──────────────────────────────────────────────────────────────────────────
    def _mainnet_execute(self, action: str, price: float, sl_pct: float, tp_pct: float) -> bool:
        """Live execution on Binance Mainnet via CCXT."""
        try:
            balance      = self.exchange.fetch_balance()
            usdt_balance = balance['total'].get('USDT', 0)

            risk_amount       = usdt_balance * config.RISK_PER_TRADE_PCT
            position_size_usd = risk_amount / sl_pct
            qty               = position_size_usd / price
            qty               = self.exchange.amount_to_precision(self.symbol, qty)

            if float(qty) <= 0:
                logger.error("Calculated quantity is too small. Aborting.")
                return False

            side = 'buy' if action == "LONG" else 'sell'
            order = self.exchange.create_market_order(self.symbol, side, qty)
            logger.info(f"[ORDER] Market {side.upper()} placed: ID={order['id']} Qty={qty}")

            sl_side  = 'sell' if action == "LONG" else 'buy'
            sl_price = float(self.exchange.price_to_precision(
                self.symbol, price * (1 - sl_pct) if action == "LONG" else price * (1 + sl_pct)))
            tp_price = float(self.exchange.price_to_precision(
                self.symbol, price * (1 + tp_pct) if action == "LONG" else price * (1 - tp_pct)))

            self.exchange.create_order(
                symbol=self.symbol, type='STOP_MARKET', side=sl_side, amount=qty, price=sl_price,
                params={'stopPrice': sl_price, 'reduceOnly': True}
            )
            self.exchange.create_order(
                symbol=self.symbol, type='TAKE_PROFIT_MARKET', side=sl_side, amount=qty, price=tp_price,
                params={'stopPrice': tp_price, 'reduceOnly': True}
            )

            logger.info(f"[SL/TP] SL={sl_price} | TP={tp_price}")
            self.trade_history.append({'action': action, 'entry': price, 'sl': sl_price, 'tp': tp_price})
            return True

        except Exception as e:
            logger.error(f"[ERROR] Mainnet execution failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    def _testnet_execute(self, action: str, price: float, sl_pct: float, tp_pct: float) -> bool:
        """Live execution on Binance Futures Testnet via raw signed HTTP requests.
        (CCXT's built-in testnet support is officially deprecated and broken.)
        """
        import requests, hmac, hashlib
        from urllib.parse import urlencode

        api_key  = config.BINANCE_API_KEY
        secret   = config.BINANCE_SECRET_KEY
        base_url = "https://testnet.binancefuture.com"

        def signed_request(method: str, endpoint: str, payload: dict = None):
            payload = payload or {}
            payload['timestamp'] = int(time.time() * 1000)
            query   = urlencode(payload)
            sig     = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            url     = f"{base_url}{endpoint}?{query}&signature={sig}"
            headers = {'X-MBX-APIKEY': api_key}
            res     = requests.get(url, headers=headers) if method == 'GET' else \
                      requests.post(url, headers=headers)
            if res.status_code != 200:
                raise Exception(f"Binance API {res.status_code}: {res.text}")
            return res.json()

        try:
            # 1. Get available balance
            account      = signed_request('GET', '/fapi/v2/account')
            usdt_balance = float(next(
                a['availableBalance'] for a in account['assets'] if a['asset'] == 'USDT'
            ))

            risk_amount       = usdt_balance * config.RISK_PER_TRADE_PCT
            position_size_usd = risk_amount / sl_pct
            qty               = round(position_size_usd / price, 3)

            if qty <= 0:
                logger.error("Calculated quantity is too small. Aborting.")
                return False

            # Symbol: 'BTC/USDT:USDT' -> 'BTCUSDT'
            raw_symbol = self.symbol.split(':')[0].replace('/', '')
            side       = 'BUY' if action == "LONG" else 'SELL'

            # 2. Market order
            order = signed_request('POST', '/fapi/v1/order', {
                'symbol': raw_symbol, 'side': side, 'type': 'MARKET', 'quantity': qty
            })
            logger.info(f"[ORDER] Market {side} placed: ID={order.get('orderId')} Qty={qty}")

            # 3. SL / TP
            sl_side  = 'SELL' if action == "LONG" else 'BUY'
            sl_price = round(price * (1 - sl_pct) if action == "LONG" else price * (1 + sl_pct), 1)
            tp_price = round(price * (1 + tp_pct) if action == "LONG" else price * (1 - tp_pct), 1)

            signed_request('POST', '/fapi/v1/order', {
                'symbol': raw_symbol, 'side': sl_side, 'type': 'STOP_MARKET',
                'quantity': qty, 'stopPrice': sl_price, 'reduceOnly': 'true'
            })
            signed_request('POST', '/fapi/v1/order', {
                'symbol': raw_symbol, 'side': sl_side, 'type': 'TAKE_PROFIT_MARKET',
                'quantity': qty, 'stopPrice': tp_price, 'reduceOnly': 'true'
            })

            logger.info(f"[SL/TP] SL={sl_price} | TP={tp_price}")
            self.trade_history.append({'action': action, 'entry': price, 'sl': sl_price, 'tp': tp_price})
            return True

        except Exception as e:
            logger.error(f"[ERROR] Testnet execution failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    def get_live_balance(self) -> float:
        """Returns the real available USDT balance from Binance (Testnet or Mainnet)."""
        if config.USE_TESTNET:
            import requests, hmac, hashlib
            from urllib.parse import urlencode
            try:
                payload = {'timestamp': int(time.time() * 1000)}
                query   = urlencode(payload)
                sig     = hmac.new(config.BINANCE_SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()
                url     = f"https://testnet.binancefuture.com/fapi/v2/account?{query}&signature={sig}"
                res     = requests.get(url, headers={'X-MBX-APIKEY': config.BINANCE_API_KEY})
                account = res.json()
                return float(next(a['availableBalance'] for a in account['assets'] if a['asset'] == 'USDT'))
            except Exception as e:
                logger.warning(f"Could not fetch live balance: {e}")
                return 0.0
        else:
            try:
                bal = self.exchange.fetch_balance()
                return float(bal.get('USDT', {}).get('total', 0.0))
            except Exception as e:
                logger.warning(f"Could not fetch live balance: {e}")
                return 0.0
