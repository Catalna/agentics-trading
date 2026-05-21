"""
core/llm_engine.py — LLM Market Context Advisor (Qwen2.5 via Ollama).

Role: Assistant only — NOT the main decision maker.
  - Analyzes market context from indicators
  - Detects conflicting conditions
  - Suggests a confidence adjustment in [-0.05, +0.05]
  - Cannot open or close positions independently

Max influence: ±5% on final confidence score.

Fails gracefully: returns neutral adjustment (0.0) on timeout or errors.
"""

from __future__ import annotations

import json
import time
import concurrent.futures
from typing import Dict, Optional

import requests

import config
from core.logger import setup_logger

log = setup_logger("LLMEngine")

# ─── Neutral fallback result ───────────────────────────────────────────────────
_NEUTRAL = {
    "market_state":         "Unknown",
    "risk_level":           "Medium",
    "confidence_adjustment": 0.0,
    "reason":               "LLM unavailable — neutral adjustment applied.",
}

# ─── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a quant trading assistant for a BTC/USDT futures scalping Expert Advisor.
Your ONLY role is to analyze the provided market context and return a JSON object.

Rules:
- You CANNOT open or close trades.
- You CANNOT override the ML model.
- You provide context and a small confidence adjustment only.
- confidence_adjustment must be in the range [-0.05, +0.05].
- Be conservative. Only adjust significantly if you detect a clear conflict or extreme condition.

Return ONLY valid JSON with these exact keys:
{
  "market_state": "<brief description>",
  "risk_level": "Low|Medium|High",
  "confidence_adjustment": <float between -0.05 and 0.05>,
  "reason": "<one sentence explanation>"
}"""


# ══════════════════════════════════════════════════════════════════════════════
# LLMEngine
# ══════════════════════════════════════════════════════════════════════════════

class LLMEngine:
    """
    Sends compressed market context to Qwen2.5 via Ollama and parses the result.

    Usage:
        llm = LLMEngine()
        result = llm.analyze(feats, ml_probs, sentiment)
        adj = result["confidence_adjustment"]   # -0.05 to +0.05
    """

    def __init__(self):
        self._available: Optional[bool] = None   # None = not yet checked
        self._last_check: float = 0.0
        self._check_interval = 300               # re-check availability every 5 min

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        feats:     Dict,
        ml_probs:  Dict,
        sentiment: Dict,
        position:  str = "NONE",
    ) -> Dict:
        """
        Build context, call LLM, parse and clamp the result.
        Returns a dict with keys: market_state, risk_level, confidence_adjustment, reason.
        """
        if not self._is_available():
            return {**_NEUTRAL, "reason": "Ollama not reachable — neutral applied."}

        # Cache to prevent spamming Ollama on every tick in high-frequency mode
        now = time.time()
        if hasattr(self, "_cached_result") and (now - getattr(self, "_cached_time", 0)) < 240:
            return self._cached_result

        prompt = self._build_prompt(feats, ml_probs, sentiment, position)
        
        multi_enabled = getattr(config, "MULTI_LLM_ENABLED", False)
        if not multi_enabled:
            raw_response = self._call_ollama(prompt, getattr(config, "OLLAMA_MODEL", "qwen2.5:7b"))
            if raw_response is None:
                return {**_NEUTRAL, "reason": "LLM call failed — neutral applied."}
            result = self._parse_response(raw_response)
            result["confidence_adjustment"] = max(
                -config.LLM_MAX_INFLUENCE,
                min(config.LLM_MAX_INFLUENCE, result["confidence_adjustment"])
            )
            log.info(
                f"[LLM] state={result['market_state']!r} "
                f"risk={result['risk_level']} "
                f"adj={result['confidence_adjustment']:+.3f} "
                f"reason={result['reason']!r}"
            )
        else:
            results = []
            def query_model(model_name, weight):
                raw = self._call_ollama(prompt, model_name)
                if raw is None: return None
                parsed = self._parse_response(raw)
                return {"model": model_name, "weight": weight, "parsed": parsed}

            models = getattr(config, "OLLAMA_MODELS", {})
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(models))) as executor:
                futures = [executor.submit(query_model, m, w) for m, w in models.items()]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res is not None:
                        results.append(res)
            
            if not results:
                return {**_NEUTRAL, "reason": "All LLMs failed — neutral applied."}

            total_weight = sum(r["weight"] for r in results)
            weighted_adj = sum(r["parsed"]["confidence_adjustment"] * r["weight"] for r in results) / total_weight if total_weight > 0 else 0.0
            weighted_adj = max(-config.LLM_MAX_INFLUENCE, min(config.LLM_MAX_INFLUENCE, weighted_adj))
            
            risk_levels = [r["parsed"]["risk_level"] for r in results]
            risk_level = max(set(risk_levels), key=risk_levels.count) if risk_levels else "Medium"
            states = [r["parsed"]["market_state"] for r in results]
            state = states[0] if states else "Unknown"
            reasons = " | ".join(f"{r['model'].split(':')[0]}: {r['parsed']['reason']}" for r in results)
            
            result = {
                "market_state": state,
                "risk_level": risk_level,
                "confidence_adjustment": weighted_adj,
                "reason": f"Multi-LLM Consensus: {reasons[:200]}"
            }
            log.info(
                f"[LLM] Multi Consensus: adj={result['confidence_adjustment']:+.3f} "
                f"from {len(results)} models. risk={result['risk_level']}"
            )
            for r in results:
                log.debug(f"[LLM] - {r['model']}: adj={r['parsed']['confidence_adjustment']:+.3f}")

        self._cached_result = result
        self._cached_time = now
        return result

    # ── Availability Check ────────────────────────────────────────────────────

    def _is_available(self) -> bool:
        now = time.time()
        if self._available is not None and (now - self._last_check) < self._check_interval:
            return self._available

        try:
            r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            self._available = True
            log.debug(f"[LLM] Ollama reachable. Models found: {len(models)}")
        except Exception as exc:
            log.warning(f"[LLM] Ollama not reachable: {exc}")
            self._available = False

        self._last_check = now
        return self._available

    # ── Prompt Builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(feats: Dict, ml_probs: Dict, sentiment: Dict, position: str) -> str:
        ctx = {
            "symbol":         "BTC/USDT",
            "timeframe":      "5m",
            "current_position": position,
            "indicators": {
                "rsi":        round(feats.get("rsi", 50), 2),
                "adx":        round(feats.get("adx", 20), 2),
                "macd_hist":  round(feats.get("macd_hist", 0), 6),
                "ema5_dist":  round(feats.get("ema5_dist", 0), 6),
                "ema20_dist": round(feats.get("ema20_dist", 0), 6),
                "vwap_dist":  round(feats.get("vwap_dist", 0), 6),
                "atr_pct":    round(feats.get("atr_pct", 0), 6),
                "volume_chg": round(feats.get("volume_change_pct", 0), 4),
            },
            "regime":     feats.get("regime", "UNKNOWN"),
            "trend_5m":   feats.get("trend_dir", "UNKNOWN"),
            "trend_15m":  feats.get("trend_15m", "UNKNOWN"),
            "ml_probs": {
                "long":  round(ml_probs.get("long_prob",  0.50), 3),
                "short": round(ml_probs.get("short_prob", 0.50), 3),
                "direction_bias": "LONG" if ml_probs.get("long_prob", 0.5) >= ml_probs.get("short_prob", 0.5) else "SHORT",
            },
            "sentiment": {
                "score":  sentiment.get("score", 0.5),
                "label":  sentiment.get("label", "NEUTRAL"),
                "extreme_event": sentiment.get("extreme_event", False),
            },
        }
        return json.dumps(ctx, separators=(",", ":"))

    # ── Ollama Call ───────────────────────────────────────────────────────────

    def _call_ollama(self, user_content: str, model_name: str = None) -> Optional[str]:
        if model_name is None:
            model_name = getattr(config, "OLLAMA_MODEL", "qwen2.5:7b")
            
        payload = {
            "model":  model_name,
            "stream": False,
            "messages": [
                {"role": "system",  "content": _SYSTEM_PROMPT},
                {"role": "user",    "content": user_content},
            ],
            "options": {
                "temperature":   0.1,       # Near-deterministic for consistency
                "num_predict":   256,       # Short response
            },
        }
        try:
            r = requests.post(
                f"{config.OLLAMA_URL}/api/chat",
                json=payload,
                timeout=config.LLM_TIMEOUT_S,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except requests.Timeout:
            log.warning(f"[LLM] Request timed out after {config.LLM_TIMEOUT_S}s")
        except Exception as exc:
            log.warning(f"[LLM] Request failed: {exc}")
        return None

    # ── Response Parser ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(text: str) -> Dict:
        """Extract JSON from LLM response, tolerate markdown code fences."""
        # Strip markdown code fences if present
        text = text.strip()
        if "```" in text:
            import re
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if m:
                text = m.group(1)

        # Find first {...} block
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

        try:
            data = json.loads(text)
            return {
                "market_state":          str(data.get("market_state", "Unknown")),
                "risk_level":            str(data.get("risk_level", "Medium")),
                "confidence_adjustment": float(data.get("confidence_adjustment", 0.0)),
                "reason":                str(data.get("reason", "")),
            }
        except (json.JSONDecodeError, ValueError) as exc:
            log.warning(f"[LLM] Parse failed: {exc}. Raw: {text[:200]}")
            return dict(_NEUTRAL)
