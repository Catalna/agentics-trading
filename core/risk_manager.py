"""
core/risk_manager.py — Dynamic Position Sizing & Circuit Breakers.

Risk tiers (based on signal confidence):
  0.75 – 0.82  → risk 0.5% of balance
  0.82 – 0.90  → risk 1.0% of balance
  > 0.90       → risk 1.5% of balance

Partial TP system:
  TP1 = ATR × 0.75  → close 50%, move SL to breakeven
  TP2 = ATR × 1.50  → close remaining 50%
  SL  = ATR × 0.50  (dynamic, never fixed dollar)

Trailing stop (after TP1):
  LONG:  new_SL = current_price − ATR × 0.30
  SHORT: new_SL = current_price + ATR × 0.30

Circuit breakers:
  Daily loss  > 2%   → halt for the day
  Drawdown    > 10%  → halt until manual reset
"""

from __future__ import annotations

import time
from typing import Tuple

import config
from core.logger import setup_logger, DBLogger

log = setup_logger("RiskManager")
db  = DBLogger()


class RiskManager:
    """
    Stateful risk controller tracking daily P&L and peak balance.

    Usage:
        rm = RiskManager(initial_balance=1000.0)
        if rm.is_trading_allowed(balance):
            qty = rm.get_position_size(balance, confidence, atr, price)
            sl, tp1, tp2 = rm.compute_sl_tp(side, price, atr)
    """

    def __init__(self, initial_balance: float = 1000.0):
        self._peak_balance    = initial_balance
        self._session_balance = initial_balance
        self._halted          = False

    # ── Public API ────────────────────────────────────────────────────────────

    def is_trading_allowed(self, balance: float) -> Tuple[bool, str]:
        """Returns (allowed, reason). Checks daily loss + drawdown."""
        if self._halted:
            return False, "Manual halt active."

        if balance > self._peak_balance:
            self._peak_balance = balance

        drawdown_pct = (self._peak_balance - balance) / max(self._peak_balance, 1e-9)
        if drawdown_pct >= config.MAX_DRAWDOWN_PCT:
            msg = (f"Max drawdown {drawdown_pct*100:.1f}% "
                   f"(limit {config.MAX_DRAWDOWN_PCT*100:.0f}%). HALTED.")
            log.warning(f"[Risk] {msg}")
            self._halted = True
            return False, msg

        daily_pnl = db.get_daily_pnl()
        if daily_pnl < 0:
            loss_pct = abs(daily_pnl) / max(self._peak_balance, 1e-9)
            if loss_pct >= config.MAX_DAILY_LOSS_PCT:
                msg = (f"Daily loss {loss_pct*100:.1f}% "
                       f"(limit {config.MAX_DAILY_LOSS_PCT*100:.0f}%). No new trades today.")
                log.warning(f"[Risk] {msg}")
                return False, msg

        return True, "OK"

    def get_position_size(
        self,
        balance:    float,
        confidence: float,
        atr:        float,
        price:      float,
    ) -> float:
        """
        Calculate position size in base asset based on dynamic margin allocation.
        This provides a stable, predictable trading size scaled by leverage,
        independent of tight ATR shifts, and guarantees the bot never exceeds
        available balance or exchange retail limits.
        """
        if price <= 0:
            log.error("[Risk] Invalid price for sizing.")
            return 0.0

        # Sizing is based on a fixed percentage of balance used as margin
        margin_pct = config.MARGIN_ALLOCATION_PCT

        # Scale size slightly based on signal confidence
        confidence_mult = 1.0
        if confidence > 0.90:
            confidence_mult = 1.2
        elif confidence < 0.75:
            confidence_mult = 0.8

        allocated_margin = balance * margin_pct * confidence_mult

        # Guarantee we never exceed 90% of available balance to prevent margin exhaustion
        if allocated_margin > balance * 0.9:
            allocated_margin = balance * 0.9

        # Target notional based on leverage
        notional = allocated_margin * config.LEVERAGE

        # Enforce maximum effective leverage cap and absolute notional cap to prevent exceeding exchange limits
        base_asset = config.SYMBOL.split("/")[0]
        max_notional = min(balance * config.MAX_EFFECTIVE_LEVERAGE, 4500.0)
        quantity = notional / price
        max_qty = max_notional / price
        if quantity > max_qty:
            quantity = max_qty
            log.warning(
                f"[Risk] Capped qty to {quantity:.4f} {base_asset} to respect exchange position limits "
                f"({config.MAX_EFFECTIVE_LEVERAGE}x, Notional Cap: ${max_notional:.2f})"
            )
        
        # Floor to minimum exchange notional
        if quantity * price < config.MIN_POSITION_NOTIONAL:
            quantity = config.MIN_POSITION_NOTIONAL / price

        quantity = round(quantity, 3)

        log.info(
            f"[Risk] balance={balance:.2f} conf={confidence:.3f} "
            f"margin_used=${allocated_margin:.2f} ({margin_pct*confidence_mult*100:.1f}%) "
            f"qty={quantity:.4f} {base_asset} (Notional: ${quantity*price:.2f})"
        )
        return quantity

    def compute_sl_tp(
        self,
        side:  str,
        price: float,
        atr:   float,
    ) -> Tuple[float, float]:
        """
        Returns (sl_price, tp_price).
        Supports both ATR-based and fixed percentage-based scalp targets.
        """
        if getattr(config, "USE_PERCENTAGE_TGT", False):
            sl_off  = price * config.SL_PCT
            tp_off  = price * config.TP_PCT
        else:
            sl_off  = atr * config.SL_ATR_MULT
            tp_off  = atr * config.TP_ATR_MULT

        # Determine precision dynamically based on asset price level
        if price < 5.0:
            p = 4
        elif price < 50.0:
            p = 3
        else:
            p = 2

        if side == "LONG":
            sl   = round(price - sl_off,  p)
            tp   = round(price + tp_off, p)
        else:  # SHORT
            sl   = round(price + sl_off,  p)
            tp   = round(price - tp_off, p)

        return sl, tp

    def halt(self):
        self._halted = True
        log.warning("[Risk] Trading manually HALTED.")

    def resume(self):
        self._halted = False
        log.info("[Risk] Trading RESUMED.")

    def update_balance(self, balance: float):
        self._session_balance = balance
        if balance > self._peak_balance:
            self._peak_balance = balance

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _risk_tier(confidence: float) -> float:
        if confidence > 0.90:
            return config.RISK_TIER_HIGH    # 1.5%
        elif confidence > 0.82:
            return config.RISK_TIER_MID     # 1.0%
        else:
            return config.RISK_TIER_LOW     # 0.5%
