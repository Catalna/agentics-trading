"""
core/signal_engine.py — Aggressive Directional Signal Engine.

Architecture:
  HOLD is NOT a prediction target. The model always answers LONG or SHORT.
  Decision is driven by comparing LONG_SCORE vs SHORT_SCORE.

Scoring (per direction):
  ML probability:       50%
  Technical conditions: 30%
  Sentiment:            10%
  LLM adjustment:       10%

Entry gates:
  No position open  → min score 0.75
  Existing position → opposite must score ≥ 0.85 to reverse
  Reversal          → opposite must score ≥ 0.90 to immediately reverse

Technical LONG bias:  EMA5>EMA20, RSI 50-70, MACD bullish, ADX>20, vol>1.2×
Technical SHORT bias: EMA5<EMA20, RSI 30-50, MACD bearish, ADX>20, vol>1.2×
"""

from __future__ import annotations

from typing import Dict

import config
from core.logger import setup_logger

log = setup_logger("SignalEngine")


# ══════════════════════════════════════════════════════════════════════════════
# Technical Scoring
# ══════════════════════════════════════════════════════════════════════════════

def _tech_score_long(feats: Dict) -> float:
    """
    Score LONG technical conditions on a [0, 1] scale.
    Based on defined entry criteria: EMA alignment, RSI zone, MACD, ADX, volume.
    """
    score = 0.0
    total = 5

    rsi           = feats.get("rsi", 50.0)
    ema5_dist     = feats.get("ema5_dist", 0.0)
    macd_hist     = feats.get("macd_hist", 0.0)
    adx           = feats.get("adx", 0.0)
    volume_ratio  = feats.get("volume_ratio", 1.0)
    macd_slope    = feats.get("macd_slope", 0.0)
    rsi_slope     = feats.get("rsi_slope", 0.0)

    # 1. EMA alignment: EMA5 > EMA20 (price above fast ema)
    if ema5_dist > 0:
        score += 1.0
    elif ema5_dist > -0.001:       # near-flat — partial credit
        score += 0.4

    # 2. RSI in bullish zone 50–70
    if config.RSI_LONG_MIN <= rsi <= config.RSI_LONG_MAX:
        score += 1.0
    elif 45 <= rsi < config.RSI_LONG_MIN:   # just below — partial
        score += 0.4
    elif rsi > config.RSI_LONG_MAX:         # overbought — negative
        score -= 0.5

    # 3. MACD bullish (histogram > 0 AND rising)
    if macd_hist > 0 and macd_slope > 0:
        score += 1.0
    elif macd_hist > 0:
        score += 0.6
    elif macd_slope > 0:           # hist negative but rising
        score += 0.3

    # 4. ADX > 20 (trending environment)
    if adx > config.ADX_TREND_THRESHOLD:
        score += 1.0
    elif adx > 15:
        score += 0.5

    # 5. Volume confirmation: ratio > 1.2×
    if volume_ratio >= config.VOLUME_RATIO_MIN:
        score += 1.0
    elif volume_ratio >= 1.0:
        score += 0.5

    return max(0.0, min(1.0, score / total))


def _tech_score_short(feats: Dict) -> float:
    """
    Score SHORT technical conditions on a [0, 1] scale.
    """
    score = 0.0
    total = 5

    rsi           = feats.get("rsi", 50.0)
    ema5_dist     = feats.get("ema5_dist", 0.0)
    macd_hist     = feats.get("macd_hist", 0.0)
    adx           = feats.get("adx", 0.0)
    volume_ratio  = feats.get("volume_ratio", 1.0)
    macd_slope    = feats.get("macd_slope", 0.0)

    # 1. EMA alignment: EMA5 < EMA20
    if ema5_dist < 0:
        score += 1.0
    elif ema5_dist < 0.001:
        score += 0.4

    # 2. RSI in bearish zone 30–50
    if config.RSI_SHORT_MIN <= rsi <= config.RSI_SHORT_MAX:
        score += 1.0
    elif config.RSI_SHORT_MAX < rsi <= 55:
        score += 0.4
    elif rsi < config.RSI_SHORT_MIN:        # oversold — negative for short
        score -= 0.5

    # 3. MACD bearish (histogram < 0 AND falling)
    if macd_hist < 0 and macd_slope < 0:
        score += 1.0
    elif macd_hist < 0:
        score += 0.6
    elif macd_slope < 0:           # hist positive but falling
        score += 0.3

    # 4. ADX > 20
    if adx > config.ADX_TREND_THRESHOLD:
        score += 1.0
    elif adx > 15:
        score += 0.5

    # 5. Volume confirmation
    if volume_ratio >= config.VOLUME_RATIO_MIN:
        score += 1.0
    elif volume_ratio >= 1.0:
        score += 0.5

    return max(0.0, min(1.0, score / total))


# ══════════════════════════════════════════════════════════════════════════════
# Composite Scoring
# ══════════════════════════════════════════════════════════════════════════════

def _composite_score(
    direction:  str,
    ml_probs:   Dict,
    feats:      Dict,
    sentiment:  Dict,
    llm_result: Dict,
    swarm_result: Dict = None,
) -> float:
    """Compute composite score for a direction in [0, 1]."""

    # ML component
    if direction == "LONG":
        ml_score   = ml_probs.get("long_prob",  0.50)
        tech_score = _tech_score_long(feats)
        sent_score = float(sentiment.get("score", 0.5))
    else:
        ml_score   = ml_probs.get("short_prob", 0.50)
        tech_score = _tech_score_short(feats)
        sent_score = 1.0 - float(sentiment.get("score", 0.5))

    # LLM adjustment
    llm_adj = float(llm_result.get("confidence_adjustment", 0.0))
    if direction == "SHORT":
        llm_adj = -llm_adj
    llm_score = max(0.0, min(1.0, 0.5 + llm_adj * 5.0))

    # Swarm adjustment
    swarm_score = 0.5
    if swarm_result:
        s_sig = swarm_result.get('signal', 'HOLD')
        s_conf = swarm_result.get('confidence', 0.0)
        if s_sig == "LONG":
            swarm_score = 0.5 + (s_conf * 0.5)
        elif s_sig == "SHORT":
            swarm_score = 0.5 - (s_conf * 0.5)
            
    if direction == "LONG":
        swarm_final = swarm_score
    else:
        swarm_final = 1.0 - swarm_score
        
    weight_swarm = getattr(config, 'WEIGHT_SWARM', 0.0)

    composite = (
        config.WEIGHT_ML        * ml_score   +
        config.WEIGHT_TECH      * tech_score +
        config.WEIGHT_SENTIMENT * sent_score +
        config.WEIGHT_LLM       * llm_score  +
        weight_swarm            * swarm_final
    )
    return round(max(0.0, min(1.0, composite)), 4)


# ══════════════════════════════════════════════════════════════════════════════
# SignalEngine
# ══════════════════════════════════════════════════════════════════════════════

class SignalEngine:
    """
    Always produces a directional signal (LONG or SHORT).
    HOLD is only returned when confidence < threshold (weak signal rejection).

    Usage:
        se = SignalEngine()
        signal = se.evaluate(ml_probs, feats, sentiment, llm_result,
                             has_position=False)
    """

    def evaluate(
        self,
        ml_probs:     Dict,
        feats:        Dict,
        sentiment:    Dict,
        llm_result:   Dict,
        swarm_result: Dict = None,
        has_position: bool = False,
        current_side: str = "NONE",
    ) -> Dict:
        """
        Evaluate and return the best directional signal.

        Returns:
            {
              "action":       "LONG" | "SHORT" | "HOLD",
              "confidence":   float,
              "long_score":   float,
              "short_score":  float,
              "breakdown":    {...},
            }
        """
        # Extreme event → block new entries
        if sentiment.get("extreme_event", False) and not has_position:
            log.warning("[Signal] EXTREME EVENT — blocking new entries.")
            return self._hold(ml_probs, feats, sentiment, llm_result, swarm_result,
                              reason="Extreme market event")

        long_score  = _composite_score("LONG",  ml_probs, feats, sentiment, llm_result, swarm_result)
        short_score = _composite_score("SHORT", ml_probs, feats, sentiment, llm_result, swarm_result)

        # Threshold depends on whether we already have a position
        threshold = config.CONF_EXISTING if has_position else config.CONF_NO_POSITION

        breakdown = {
            "long_score":      long_score,
            "short_score":     short_score,
            "threshold":       threshold,
            "ml_long":         round(ml_probs.get("long_prob",  0.5), 4),
            "ml_short":        round(ml_probs.get("short_prob", 0.5), 4),
            "tech_long":       round(_tech_score_long(feats),  4),
            "tech_short":      round(_tech_score_short(feats), 4),
            "sentiment_score": round(float(sentiment.get("score", 0.5)), 4),
            "sentiment_label": sentiment.get("label", "NEUTRAL"),
            "llm_adj":         round(float(llm_result.get("confidence_adjustment", 0.0)), 4),
            "swarm_signal":    swarm_result.get('signal', 'HOLD') if swarm_result else 'HOLD',
            "swarm_conf":      round(float(swarm_result.get('confidence', 0.0)), 4) if swarm_result else 0.0,
            "regime":          feats.get("regime", ""),
            "trend_5m":        feats.get("trend_dir", ""),
            "trend_15m":       feats.get("trend_15m", ""),
            "volume_ratio":    round(feats.get("volume_ratio", 1.0), 3),
            "rsi":             round(feats.get("rsi", 50.0), 2),
            "adx":             round(feats.get("adx", 0.0), 2),
        }

        best_score     = max(long_score, short_score)
        best_direction = "LONG" if long_score >= short_score else "SHORT"

        if has_position and best_direction == current_side:
            log.debug(
                f"[Signal] HOLD — suppressed same-direction {best_direction} signal "
                f"score={best_score:.3f} while already in position."
            )
            return self._hold(ml_probs, feats, sentiment, llm_result, swarm_result,
                              reason=f"Already {current_side}",
                              long_score=long_score, short_score=short_score)

        if best_score >= threshold:
            log.info(
                f"[Signal] {best_direction} score={best_score:.3f} "
                f"(vs {'SHORT' if best_direction=='LONG' else 'LONG'}={min(long_score,short_score):.3f}) "
                f"threshold={threshold:.2f}"
            )
            return {
                "action":      best_direction,
                "confidence":  best_score,
                "long_score":  long_score,
                "short_score": short_score,
                "breakdown":   breakdown,
            }

        log.debug(
            f"[Signal] HOLD — best={best_score:.3f} < threshold={threshold:.2f} "
            f"(long={long_score:.3f} short={short_score:.3f})"
        )
        return self._hold(ml_probs, feats, sentiment, llm_result, swarm_result,
                          long_score=long_score, short_score=short_score)

    @staticmethod
    def _hold(ml_probs, feats, sentiment, llm_result, swarm_result=None,
              reason="", long_score=0.0, short_score=0.0) -> Dict:
        return {
            "action":      "HOLD",
            "confidence":  0.0,
            "long_score":  long_score,
            "short_score": short_score,
            "reason":      reason,
            "breakdown": {
                "long_score":      long_score,
                "short_score":     short_score,
                "threshold":       0.0,
                "ml_long":         round(ml_probs.get("long_prob",  0.5), 4),
                "ml_short":        round(ml_probs.get("short_prob", 0.5), 4),
                "tech_long":       round(_tech_score_long(feats),  4),
                "tech_short":      round(_tech_score_short(feats), 4),
                "sentiment_score": round(float(sentiment.get("score", 0.5)), 4),
                "sentiment_label": sentiment.get("label", "NEUTRAL"),
                "llm_adj":         round(float(llm_result.get("confidence_adjustment", 0.0)), 4),
                "swarm_signal":    swarm_result.get('signal', 'HOLD') if swarm_result else 'HOLD',
                "swarm_conf":      round(float(swarm_result.get('confidence', 0.0)), 4) if swarm_result else 0.0,
                "regime":          feats.get("regime", ""),
                "trend_5m":        feats.get("trend_dir", ""),
                "trend_15m":       feats.get("trend_15m", ""),
                "volume_ratio":    round(feats.get("volume_ratio", 1.0), 3),
                "rsi":             round(feats.get("rsi", 50.0), 2),
                "adx":             round(feats.get("adx", 0.0), 2),
            },
        }
