"""
core/trade_monitor.py — Active Trade Surveillance (Monitor Mode).

Runs on every newly closed 5m candle while a position is open.

Multi-factor hold scoring:
  Score each factor 0 or 1 (or fractional). Total max = 6 points.
  If score < 3 → consider closing.

Monitored factors:
  1. EMA direction maintained
  2. Momentum (MACD histogram direction & slope)
  3. RSI change (moving with or against position)
  4. ATR stability (not expanding against us)
  5. Volume strength (volume_ratio ≥ 1.0)
  6. No strong opposite signal

Hold duration philosophy:
  Strong trend (ADX>30):   15–45 min → allow longer hold
  Moderate (ADX 20–30):    5–20 min  → normal duration
  Weak (ADX<20):           1–5 min   → exit fast

Trailing stop:
  Active after TP1 hit.
  LONG:  SL = current_price − ATR×0.30
  SHORT: SL = current_price + ATR×0.30
"""

from __future__ import annotations

from typing import Dict, Optional

import config
from core.logger import setup_logger

log = setup_logger("TradeMonitor")


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Factor Hold Scorer
# ══════════════════════════════════════════════════════════════════════════════

def _hold_score(state: str, feats: Dict, signal: Dict) -> float:
    """
    Score the current position's health on [0, 6] scale.
    Returns a score from 0 (exit immediately) to 6 (hold strongly).
    """
    score = 0.0

    ema5_dist   = feats.get("ema5_dist",  0.0)
    macd_hist   = feats.get("macd_hist",  0.0)
    macd_slope  = feats.get("macd_slope", 0.0)
    rsi         = feats.get("rsi",        50.0)
    rsi_slope   = feats.get("rsi_slope",  0.0)
    atr_pct     = feats.get("atr_pct",    0.0)
    vol_ratio   = feats.get("volume_ratio", 1.0)

    opp_action  = signal.get("action", "HOLD")
    opp_conf    = signal.get("confidence", 0.0)
    opp_score   = signal.get("short_score" if state == "LONG" else "long_score", 0.0)

    if state == "LONG":
        # 1. EMA alignment
        if ema5_dist > 0:
            score += 1.0
        elif ema5_dist > -0.0005:
            score += 0.5

        # 2. MACD momentum
        if macd_hist > 0 and macd_slope >= 0:
            score += 1.0
        elif macd_hist > 0:
            score += 0.5

        # 3. RSI direction
        if 50 <= rsi <= 75 and rsi_slope >= 0:
            score += 1.0
        elif 45 <= rsi < 50:
            score += 0.4
        elif rsi > 75:
            score += 0.2   # overbought — partial

        # 4. ATR stability (not expanding >1.5% of price)
        if atr_pct < 0.015:
            score += 1.0
        elif atr_pct < 0.025:
            score += 0.5

        # 5. Volume support
        if vol_ratio >= config.VOLUME_RATIO_MIN:
            score += 1.0
        elif vol_ratio >= 1.0:
            score += 0.5

        # 6. No strong opposite signal
        if opp_action != "SHORT" or opp_conf < config.CONF_EXISTING:
            score += 1.0
        elif opp_conf < config.CONF_REVERSAL:
            score += 0.3

    else:   # SHORT
        # 1. EMA alignment
        if ema5_dist < 0:
            score += 1.0
        elif ema5_dist < 0.0005:
            score += 0.5

        # 2. MACD momentum
        if macd_hist < 0 and macd_slope <= 0:
            score += 1.0
        elif macd_hist < 0:
            score += 0.5

        # 3. RSI direction
        if 25 <= rsi <= 50 and rsi_slope <= 0:
            score += 1.0
        elif 50 < rsi <= 55:
            score += 0.4
        elif rsi < 25:
            score += 0.2   # oversold — partial

        # 4. ATR stability
        if atr_pct < 0.015:
            score += 1.0
        elif atr_pct < 0.025:
            score += 0.5

        # 5. Volume support
        if vol_ratio >= config.VOLUME_RATIO_MIN:
            score += 1.0
        elif vol_ratio >= 1.0:
            score += 0.5

        # 6. No strong opposite signal
        if opp_action != "LONG" or opp_conf < config.CONF_EXISTING:
            score += 1.0
        elif opp_conf < config.CONF_REVERSAL:
            score += 0.3

    return round(score, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic Hold Duration
# ══════════════════════════════════════════════════════════════════════════════

def _max_hold_minutes(feats: Dict) -> int:
    """Determine max holding time based on ADX (trend strength)."""
    adx = feats.get("adx", 20.0)
    if adx > 30:
        return 45   # Strong trend — allow up to 45 min
    elif adx > config.ADX_TREND_THRESHOLD:
        return 20   # Moderate trend — 20 min
    else:
        return 5    # Weak/no trend — exit quickly


# ══════════════════════════════════════════════════════════════════════════════
# TradeMonitor
# ══════════════════════════════════════════════════════════════════════════════

class TradeMonitor:
    """
    Monitors an active position and decides: HOLD | CLOSE_EARLY | REVERSE.

    Usage:
        monitor = TradeMonitor()
        monitor.reset(sl_price)                    # called on new trade open
        action = monitor.check(feats, state, signal, exec_eng, pm, open_ts)
    """

    def __init__(self):
        self._current_sl: float = 0.0

    def reset(self, sl_price: float):
        """Called when a new position opens."""
        self._current_sl      = sl_price

    def check(
        self,
        feats:    Dict,
        state:    str,
        signal:   Dict,
        exec_eng,
        pm,
        open_ts:  float,
    ) -> str:
        """
        Evaluate every candle while a position is open.
        Returns: HOLD | CLOSE_EARLY | REVERSE
        """
        if state == "NONE":
            return "HOLD"

        import time
        duration_min = (time.time() - open_ts) / 60

        close = feats.get("close", 0.0)
        atr   = feats.get("atr",   0.0)
        adx   = feats.get("adx",   0.0)

        # ── 1. Dynamic max hold duration exceeded ────────────────────────────
        max_hold = _max_hold_minutes(feats)
        if duration_min > max_hold:
            log.info(
                f"[Monitor] Max hold duration exceeded "
                f"({duration_min:.1f}m > {max_hold}m ADX={adx:.1f}) — closing."
            )
            pm.close_trade(reason="max_duration")
            self._current_sl = 0.0
            return "CLOSE_EARLY"

        # ── 2. Multi-factor hold scoring ──────────────────────────────────────
        score = _hold_score(state, feats, signal)
        log.debug(
            f"[Monitor] Hold score={score:.1f}/6 "
            f"dur={duration_min:.1f}m state={state}"
        )

        # Exit if health score drops below threshold (hyper-sensitive HFT threshold)
        if score < 3.5:
            log.info(
                f"[Monitor] Weak hold score {score:.1f}/6 — closing early. "
                f"dur={duration_min:.1f}m"
            )
            pm.close_trade(reason="momentum_loss")
            self._current_sl = 0.0
            return "CLOSE_EARLY"

        # ── 3. Reversal signal check ──────────────────────────────────────────
        sig_action = signal.get("action", "HOLD")
        sig_conf   = signal.get("confidence", 0.0)

        if state == "LONG" and sig_action == "SHORT" and sig_conf >= config.CONF_REVERSAL:
            log.info(f"[Monitor] REVERSAL signal SHORT conf={sig_conf:.3f} — flagging.")
            return "REVERSE"
        if state == "SHORT" and sig_action == "LONG" and sig_conf >= config.CONF_REVERSAL:
            log.info(f"[Monitor] REVERSAL signal LONG conf={sig_conf:.3f} — flagging.")
            return "REVERSE"

        return "HOLD"
