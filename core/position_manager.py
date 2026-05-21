"""
core/position_manager.py — Aggressive Position State Machine.

States: NONE | LONG | SHORT

New architecture:
  NONE state  → always enter LONG or SHORT if signal clears threshold.
  Active state → Monitor Mode. Only action if:
    - TP1/TP2 hit (handled by ExecutionEngine)
    - Momentum loss detected (TradeMonitor)
    - Opposite signal ≥ 0.90 confidence (reversal)

HOLD is only returned when in cooldown or risk check fails.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import config
from core.logger import setup_logger, DBLogger
from core.risk_manager import RiskManager
from core.execution_engine import ExecutionEngine

log = setup_logger("PositionManager")
db  = DBLogger()


class PositionManager:
    """
    Routes directional signals to the execution engine.

    Usage:
        pm = PositionManager(exec_engine, risk_manager)
        result = pm.process_signal(signal, feats, balance)
    """

    def __init__(self, exec_engine: ExecutionEngine, risk_manager: RiskManager):
        self._exec   = exec_engine
        self._risk   = risk_manager
        self._state  = "NONE"               # NONE | LONG | SHORT
        self._trade_id: Optional[str] = None
        self._open_ts:  float         = 0.0
        self._sl_price: float         = 0.0
        self._tp_price: float         = 0.0
        self._candles_since_close: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def trade_id(self) -> Optional[str]:
        return self._trade_id

    @property
    def open_ts(self) -> float:
        return self._open_ts

    @property
    def sl_price(self) -> float:
        return self._sl_price

    @property
    def tp_price(self) -> float:
        return self._tp_price

    def process_signal(
        self,
        signal:  Dict,
        feats:   Dict,
        balance: float,
    ) -> str:
        """
        Route signal to the correct action based on current state.

        Returns action string:
          OPEN_LONG | OPEN_SHORT | HOLD | REVERSE_LONG_TO_SHORT |
          REVERSE_SHORT_TO_LONG | CLOSE
        """
        action     = signal.get("action", "HOLD")
        confidence = signal.get("confidence", 0.0)

        # Cooldown enforcement
        if self._state == "NONE" and self._candles_since_close < config.COOLDOWN_AFTER_TRADE:
            self._candles_since_close += 1
            log.debug(f"[PM] Cooldown: {self._candles_since_close}/{config.COOLDOWN_AFTER_TRADE}")
            return "HOLD"

        # ── No position ────────────────────────────────────────────────────────
        if self._state == "NONE":
            if action == "HOLD":
                return "HOLD"
            return self._open(action, feats, balance, confidence)

        # ── Position active: Monitor Mode ──────────────────────────────────────
        # Only a STRONG opposite signal triggers a reversal
        if action == "HOLD":
            return "MONITOR"

        if action != self._state:
            # Opposite direction — check reversal threshold
            if confidence >= config.CONF_REVERSAL:
                log.info(
                    f"[PM] REVERSAL triggered: "
                    f"{self._state}→{action} conf={confidence:.3f}"
                )
                return self._reverse(action, feats, balance, confidence)
            else:
                log.debug(
                    f"[PM] Opposite signal ({action}) conf={confidence:.3f} "
                    f"< reversal threshold {config.CONF_REVERSAL:.2f} — holding."
                )
                return "MONITOR"

        # Same direction as current position — keep monitoring
        return "MONITOR"

    def close_trade(self, reason: str = "signal") -> Optional[float]:
        """External close trigger (from TradeMonitor)."""
        if self._state == "NONE":
            return None
        exit_price = self._exec.close_position(reason=reason)
        if exit_price is not None:
            self._record_close(exit_price, reason)
        return exit_price

    def update_sl(self, new_sl: float):
        """Update tracked SL price (called after trailing stop or BE move)."""
        self._sl_price = new_sl

    def sync_state_with_exchange(self):
        """
        Query the exchange for active positions and synchronize local state.
        This runs on startup to prevent blind trading if a position was left open.
        """
        try:
            positions = self._exec.get_open_positions()
            if positions:
                pos = positions[0]  # get_open_positions already filters for SYMBOL
                contracts = float(pos.get("contracts", 0.0))
                entry_price = float(pos.get("entryPrice", 0.0))
                
                if contracts > 0:
                    self._state = "LONG"
                    self._exec._current_side = "LONG"
                    self._exec._quantity = contracts
                    self._exec._entry_price = entry_price
                    log.info(f"[Sync] Detected active LONG position on Binance: {contracts} contracts @ {entry_price:.2f}")
                elif contracts < 0:
                    self._state = "SHORT"
                    self._exec._current_side = "SHORT"
                    self._exec._quantity = abs(contracts)
                    self._exec._entry_price = entry_price
                    log.info(f"[Sync] Detected active SHORT position on Binance: {abs(contracts)} contracts @ {entry_price:.2f}")
            else:
                self._state = "NONE"
                self._exec._current_side = "NONE"
                log.info("[Sync] No active positions detected on Binance. Local state synchronized to NONE.")
        except Exception as exc:
            log.error(f"[Sync] Failed to synchronize state with Binance: {exc}")

    # ── Internal Transitions ──────────────────────────────────────────────────

    def _open(
        self,
        side:       str,
        feats:      Dict,
        balance:    float,
        confidence: float,
    ) -> str:
        atr   = feats.get("atr",   0.0)
        price = feats.get("close", 0.0)

        allowed, reason = self._risk.is_trading_allowed(balance)
        if not allowed:
            log.warning(f"[PM] Blocked: {reason}")
            return "HOLD"

        qty = self._risk.get_position_size(balance, confidence, atr, price)
        if qty <= 0:
            log.warning("[PM] Zero quantity — skipping.")
            return "HOLD"

        sl, tp = self._risk.compute_sl_tp(side, price, atr)

        trade_id = self._exec.open_position(side, qty, price, sl, tp)
        if trade_id is None:
            return "HOLD"

        self._state      = side
        self._trade_id   = trade_id
        self._open_ts    = time.time()
        self._sl_price   = sl
        self._tp_price   = tp
        self._candles_since_close = 0

        db.open_trade(
            trade_id=trade_id,
            symbol=config.SYMBOL,
            side=side,
            entry_price=price,
            quantity=qty,
            sl_price=sl,
            tp_price=tp,
            confidence=confidence,
        )
        fmt = ".4f" if price < 5.0 else (".3f" if price < 50.0 else ".2f")
        log.info(
            f"[PM] OPENED {side} | "
            f"qty={qty} price={price:{fmt}} "
            f"SL={sl:{fmt}} TP={tp:{fmt}}"
        )
        return f"OPEN_{side}"

    def _reverse(
        self,
        new_side:   str,
        feats:      Dict,
        balance:    float,
        confidence: float,
    ) -> str:
        old_side = self._state
        atr      = feats.get("atr",   0.0)
        price    = feats.get("close", 0.0)

        allowed, reason = self._risk.is_trading_allowed(balance)
        if not allowed:
            log.warning(f"[PM] Reverse blocked: {reason} — closing only.")
            self.close_trade(reason="risk_limit_on_reverse")
            return "CLOSE"

        qty = self._risk.get_position_size(balance, confidence, atr, price)
        sl, tp = self._risk.compute_sl_tp(new_side, price, atr)

        exit_price = self._exec.close_position(reason=f"reverse_to_{new_side}")
        if exit_price:
            self._record_close(exit_price, f"reverse_to_{new_side}")

        if qty > 0:
            trade_id = self._exec.open_position(new_side, qty, price, sl, tp)
            if trade_id:
                self._state      = new_side
                self._trade_id   = trade_id
                self._open_ts    = time.time()
                self._sl_price   = sl
                self._tp_price   = tp
                self._candles_since_close = 0

                db.open_trade(
                    trade_id=trade_id,
                    symbol=config.SYMBOL,
                    side=new_side,
                    entry_price=price,
                    quantity=qty,
                    sl_price=sl,
                    tp_price=tp,
                    confidence=confidence,
                )
                fmt = ".4f" if price < 5.0 else (".3f" if price < 50.0 else ".2f")
                log.info(f"[PM] REVERSED {old_side}→{new_side} @ {price:{fmt}}")
                return f"REVERSE_{old_side}_TO_{new_side}"

        self._state = "NONE"
        return "CLOSE"

    def _record_close(self, exit_price: float, reason: str):
        """Calculate P&L and log the trade closure."""
        if not self._trade_id:
            return

        entry = self._exec.entry_price or exit_price
        qty   = self._exec.quantity
        side  = self._state

        if side == "LONG":
            pnl_usdt = (exit_price - entry) * qty * config.LEVERAGE
        elif side == "SHORT":
            pnl_usdt = (entry - exit_price) * qty * config.LEVERAGE
        else:
            pnl_usdt = 0.0

        pnl_pct      = pnl_usdt / max(entry * qty, 1e-9) * 100
        duration_min = (time.time() - self._open_ts) / 60

        db.close_trade(
            trade_id=self._trade_id,
            exit_price=exit_price,
            pnl_usdt=round(pnl_usdt, 4),
            pnl_pct=round(pnl_pct, 4),
            duration_min=round(duration_min, 2),
            exit_reason=reason,
        )
        log.info(
            f"[PM] Trade {self._trade_id} CLOSED | "
            f"PnL={pnl_usdt:+.2f} USDT ({pnl_pct:+.2f}%) "
            f"dur={duration_min:.1f}m reason={reason}"
        )
        self._state    = "NONE"
        self._trade_id = None
        self._open_ts  = 0.0
        self._candles_since_close = 0
