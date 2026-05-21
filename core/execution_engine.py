"""
core/execution_engine.py — Binance Futures Order Execution via ccxt.

Supports:
  open_position(side, qty, sl, tp1, tp2)   → market entry + SL + TP1 orders
  partial_close(qty_to_close, reason)      → close 50% at TP1, move SL to BE
  close_position(reason)                   → full market close
  reverse_position(new_side, qty, sl, tp1, tp2)
  update_trailing_stop(new_sl)
  set_leverage()
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional, Tuple

import ccxt

import config
from core.logger import setup_logger

log = setup_logger("Execution")


class ExecutionEngine:
    """
    ccxt wrapper for Binance USDT-M Futures with partial TP support.
    """

    def __init__(self):
        self.exchange        = self._build_exchange()
        self._symbol         = config.SYMBOL
        self._leverage       = config.LEVERAGE
        self._sl_order_id:   Optional[str] = None
        self._tp_order_id:   Optional[str] = None
        self._entry_price:   float = 0.0
        self._current_side:  str   = "NONE"
        self._quantity:      float = 0.0

    # ── Exchange Setup ────────────────────────────────────────────────────────

    @staticmethod
    def _build_exchange() -> ccxt.Exchange:
        ex = ccxt.binanceusdm({
            "apiKey":  config.API_KEY,
            "secret":  config.API_SECRET,
            "options": {"defaultType": "future"},
            "enableRateLimit": True,
            "verify": False,  # Bypass SSL certificate errors on Windows
        })
        if config.TESTNET:
            try:
                ex.enable_demo_trading(True)
                log.info("[Exec] Running against Binance Futures DEMO TRADING")
            except Exception as e:
                log.error(f"[Exec] Failed to enable demo trading: {e}")
        else:
            log.info("[Exec] Running against Binance Futures LIVE")
        
        return ex

    def set_leverage(self):
        try:
            self.exchange.set_leverage(self._leverage, self._symbol)
            log.info(f"[Exec] Leverage set to {self._leverage}x for {self._symbol}")
        except Exception as exc:
            log.warning(f"[Exec] set_leverage failed (using default browser interface leverage): {exc}")

    # ── Balance & Positions ───────────────────────────────────────────────────

    def get_balance(self) -> float:
        try:
            bal = self.exchange.fetch_balance()
            return float(bal.get("USDT", {}).get("total", 0.0))
        except Exception as exc:
            log.error(f"[Exec] get_balance error: {exc}")
            return 0.0

    def get_open_positions(self) -> List[Dict]:
        try:
            positions = self.exchange.fetch_positions([self._symbol])
            return [p for p in positions if float(p.get("contracts", 0)) != 0]
        except Exception as exc:
            log.error(f"[Exec] get_open_positions error: {exc}")
            return []

    def has_open_position(self) -> bool:
        return bool(self.get_open_positions())

    # ── Order Placement ───────────────────────────────────────────────────────

    def open_position(
        self,
        side:      str,
        quantity:  float,
        price:     float,  # Candle close price
        sl_price:  float,
        tp_price:  float,
    ) -> Optional[str]:
        """
        Open a new position with market/limit entry + SL + TP (100%).
        Returns trade_id or None on failure.
        """
        if self._current_side != "NONE":
            log.warning(f"[Exec] Cannot open {side}: already {self._current_side}")
            return None

        ccxt_side  = "buy"  if side == "LONG" else "sell"
        close_side = "sell" if side == "LONG" else "buy"
        entry_order_id = None

        try:
            # 1. Fetch current live bid/ask from orderbook or ticker to place the limit order exactly at market edge
            live_price = price
            try:
                ob = self.exchange.fetch_order_book(self._symbol, limit=1)
                live_price = float(ob['asks'][0][0] if ccxt_side == 'buy' else ob['bids'][0][0])
            except Exception:
                try:
                    ticker = self.exchange.fetch_ticker(self._symbol)
                    live_price = float(ticker.get('ask') or ticker.get('bid') or ticker.get('last') or ticker.get('close') or price)
                except Exception:
                    pass

            if live_price <= 0:
                live_price = price

            # Adjust SL/TP levels relative to the actual limit entry price to prevent -2021 trigger issues
            price_diff = live_price - price
            sl_price   += price_diff
            tp_price   += price_diff

            # 2. Place entry order depending on config
            order_type = getattr(config, "ORDER_TYPE", "market")
            if order_type == "market":
                log.info(f"[Exec] Placing MARKET entry order for {side}...")
                entry_order = self.exchange.create_order(
                    self._symbol, "market", ccxt_side, quantity,
                    params={"reduceOnly": False}
                )
                entry_order_id = entry_order.get("id")
                # Retrieve execution price if returned by CCXT, otherwise fallback to live_price
                avg_price = float(entry_order.get("average") or entry_order.get("price") or 0)
                if avg_price > 0:
                    live_price = avg_price
            else:
                log.info(f"[Exec] Placing LIMIT entry order for {side} at {live_price:.2f}...")
                entry_order = self.exchange.create_order(
                    self._symbol, "limit", ccxt_side, quantity, live_price,
                    params={"reduceOnly": False}
                )
                entry_order_id = entry_order.get("id")

            self._entry_price = live_price
            self._current_side = side
            self._quantity     = quantity
            trade_id           = str(uuid.uuid4())[:12]

            fmt = ".4f" if self._entry_price < 5.0 else (".3f" if self._entry_price < 50.0 else ".2f")
            log.info(
                f"[Exec] OPENED {side} | qty={quantity} "
                f"entry={self._entry_price:{fmt}} SL={sl_price:{fmt}} "
                f"TP={tp_price:{fmt}}"
            )

            # 3. Stop Loss (full quantity)
            sl_order = self.exchange.create_order(
                self._symbol, "stop_market", close_side, quantity,
                params={"stopPrice": sl_price, "reduceOnly": True,
                        "workingType": "CONTRACT_PRICE"}
            )
            self._sl_order_id = sl_order.get("id")

            # 4. Take Profit (full quantity)
            tp_order = self.exchange.create_order(
                self._symbol, "take_profit_market", close_side, quantity,
                params={"stopPrice": tp_price, "reduceOnly": True,
                        "workingType": "CONTRACT_PRICE"}
            )
            self._tp_order_id = tp_order.get("id")

            return trade_id

        except ccxt.InsufficientFunds as exc:
            log.error(f"[Exec] Insufficient funds: {exc}")
            self._reset_state()
        except ccxt.ExchangeError as exc:
            log.error(f"[Exec] Exchange error on open: {exc}")
            # Rollback entry order if SL/TP fails to prevent unprotected fills
            if entry_order_id:
                try:
                    self.exchange.cancel_order(entry_order_id, self._symbol)
                    log.info(f"[Exec] Safely cancelled pending entry order {entry_order_id} due to SL/TP setup failure.")
                except Exception as cancel_exc:
                    log.warning(f"[Exec] Could not cancel entry order {entry_order_id}: {cancel_exc}")
            self._reset_state()
        except Exception as exc:
            log.error(f"[Exec] Unexpected error on open: {exc}", exc_info=True)
            self._reset_state()
        return None

    def close_position(self, reason: str = "signal") -> Optional[float]:
        """Full market close of current position."""
        if self._current_side == "NONE":
            return None

        self._cancel_all_orders()

        close_side = "sell" if self._current_side == "LONG" else "buy"
        qty = self._quantity
        exit_price = None
        try:
            # Market order exit for instant filling (essential for HFT)
            order = self.exchange.create_order(
                self._symbol, "market", close_side, qty,
                params={"reduceOnly": True}
            )
            exit_price = float(order.get("average") or order.get("price") or 0)
            if exit_price == 0:
                try:
                    ticker = self.exchange.fetch_ticker(self._symbol)
                    exit_price = float(ticker.get('last') or ticker.get('close') or 0)
                except Exception:
                    pass
            log.info(
                f"[Exec] CLOSED {self._current_side} | "
                f"entry={self._entry_price:.2f} exit={exit_price:.2f} reason={reason}"
            )
            self._reset_state()
            return exit_price
        except ccxt.ExchangeError as exc:
            err_msg = str(exc)
            if "-2022" in err_msg or "ReduceOnly Order is rejected" in err_msg:
                log.info(f"[Exec] Limit entry was never filled (no position to close). Cleanly resetting state.")
                self._reset_state()
            else:
                log.error(f"[Exec] Exchange error on close: {exc}")
        except Exception as exc:
            log.error(f"[Exec] Unexpected error on close: {exc}", exc_info=True)
        return None

    def reverse_position(
        self,
        new_side:  str,
        quantity:  float,
        price:     float,  # Limit entry price
        sl_price:  float,
        tp_price:  float,
    ) -> Optional[str]:
        """Close current position then open opposite."""
        log.info(f"[Exec] REVERSING {self._current_side} → {new_side}")
        self.close_position(reason=f"reversal_to_{new_side}")
        time.sleep(0.5)
        return self.open_position(new_side, quantity, price, sl_price, tp_price)

    def update_trailing_stop(self, new_sl_price: float) -> bool:
        """Replace existing SL with a tighter trailing stop."""
        if self._current_side == "NONE":
            return False

        self._cancel_order(self._sl_order_id)
        close_side = "sell" if self._current_side == "LONG" else "buy"
        try:
            sl_order = self.exchange.create_order(
                self._symbol, "stop_market", close_side, self._quantity,
                params={"stopPrice": new_sl_price, "reduceOnly": True,
                        "workingType": "CONTRACT_PRICE"}
            )
            self._sl_order_id = sl_order.get("id")
            log.info(f"[Exec] Trailing SL → {new_sl_price:.2f}")
            return True
        except Exception as exc:
            log.error(f"[Exec] update_trailing_stop error: {exc}")
            return False

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _check_order_filled(self, order_id: Optional[str]) -> bool:
        """Check if a given order_id has been filled."""
        if not order_id:
            return False
        try:
            order = self.exchange.fetch_order(order_id, self._symbol)
            return order.get("status") in ("closed", "filled")
        except Exception:
            return False

    def _cancel_all_orders(self):
        for oid in (self._sl_order_id, self._tp_order_id):
            self._cancel_order(oid)
        self._sl_order_id  = None
        self._tp_order_id  = None

    def _cancel_order(self, order_id: Optional[str]):
        if not order_id:
            return
        try:
            self.exchange.cancel_order(order_id, self._symbol)
            log.debug(f"[Exec] Cancelled order {order_id}")
        except ccxt.OrderNotFound:
            pass
        except Exception as exc:
            log.warning(f"[Exec] Cancel {order_id}: {exc}")

    def _reset_state(self):
        self._current_side = "NONE"
        self._quantity     = 0.0
        self._entry_price  = 0.0
        self._sl_order_id  = None
        self._tp_order_id  = None

    @property
    def current_side(self) -> str:
        return self._current_side

    @property
    def entry_price(self) -> float:
        return self._entry_price

    @property
    def quantity(self) -> float:
        return self._quantity
