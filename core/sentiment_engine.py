"""
core/sentiment_engine.py — Aggregates sentiment from pure API/RSS sources.

Sources (7 total):
  1. Fear & Greed Index (API)
  2. CoinTelegraph (RSS)
  3. Decrypt (RSS)
  4. Binance Announcements (RSS)
  5. Bitcoin.com (RSS)
  6. CryptoPotato (RSS)
  7. YouTube (RSS)

Outputs a normalized [0, 1] score and a label.
Runs asynchronously, updating a JSON cache file.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import config
from core.logger import setup_logger

log = setup_logger("Sentiment")


class SentimentEngine:
    """
    Background worker that fetches and caches market sentiment.

    Usage:
        se = SentimentEngine()
        se.start()
        ...
        sent = se.get_sentiment()  # Returns dict immediately from cache
    """

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self._running = False
        self._threads: List[threading.Thread] = []
        self._cache_lock = threading.Lock()
        self._state = {
            "sources": {
                "fng":           {"score": 0.5, "updated": 0},
                "cointelegraph": {"score": 0.5, "updated": 0},
                "decrypt":       {"score": 0.5, "updated": 0},
                "binance_ann":   {"score": 0.5, "updated": 0},
                "bitcoincom":    {"score": 0.5, "updated": 0},
                "cryptopotato":  {"score": 0.5, "updated": 0},
            },
            "composite": {
                "score": 0.5,
                "label": "NEUTRAL",
                "extreme_event": False,
                "updated": 0,
            }
        }
        self._load_cache()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start all background fetching threads."""
        if self._running:
            return
        self._running = True

        log.info("[Sentiment] Starting background refresh threads…")

        def _launch(target, kwargs):
            t = threading.Thread(target=target, kwargs=kwargs, daemon=True)
            self._threads.append(t)
            t.start()

        _launch(self._fetch_fng,     {"interval": config.SENT_REFRESH_FNG,     "initial_delay": 1})
        _launch(self._fetch_rss,     {"name": "cointelegraph", "url": config.COINTELEGRAPH_RSS, "interval": config.SENT_REFRESH_RSS, "initial_delay": 1})
        _launch(self._fetch_rss,     {"name": "decrypt",       "url": config.DECRYPT_RSS,       "interval": config.SENT_REFRESH_RSS, "initial_delay": 2})
        _launch(self._fetch_rss,     {"name": "binance_ann",   "url": config.BINANCE_ANN_RSS,   "interval": config.SENT_REFRESH_BINANCE, "initial_delay": 3})
        _launch(self._fetch_rss,     {"name": "bitcoincom",    "url": config.BITCOINCOM_RSS,    "interval": config.SENT_REFRESH_RSS, "initial_delay": 4})
        _launch(self._fetch_rss,     {"name": "cryptopotato",  "url": config.CRYPTOPOTATO_RSS,  "interval": config.SENT_REFRESH_RSS, "initial_delay": 5})

        _launch(self._aggregate_loop, {"interval": 30, "initial_delay": 10})

    def stop(self):
        """Stop threads gracefully."""
        self._running = False
        log.info("[Sentiment] Stopped.")

    def get_sentiment(self) -> Dict:
        """Return the latest composite sentiment dict."""
        with self._cache_lock:
            return self._state["composite"].copy()

    # ── Fetchers ──────────────────────────────────────────────────────────────

    def _fetch_fng(self, interval: int, initial_delay: int):
        time.sleep(initial_delay)
        log.debug(f"[Sentiment/FNG] Thread started (interval={interval}s, initial={initial_delay}s)")
        while self._running:
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                resp = requests.get(config.FNG_URL, headers=headers, verify=False, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if data.get("data"):
                    fng_val = float(data["data"][0]["value"])
                    label   = data["data"][0]["value_classification"]
                    score   = fng_val / 100.0
                    self._update_source("fng", score)
                    log.debug(f"[FNG] score={score:.3f} label={label}")
            except Exception as exc:
                log.warning(f"[Sentiment/FNG] Fetch failed: {exc}")
            self._sleep(interval)

    def _fetch_rss(self, name: str, url: str, interval: int, initial_delay: int):
        time.sleep(initial_delay)
        log.debug(f"[Sentiment/{name}] Thread started (interval={interval}s, initial={initial_delay}s)")
        while self._running:
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                resp = requests.get(url, headers=headers, verify=False, timeout=10)
                resp.raise_for_status()
                feed = feedparser.parse(resp.text)
                texts = []
                for entry in feed.entries[:25]:
                    title   = getattr(entry, "title", "")
                    summary = getattr(entry, "summary", "")
                    texts.append(f"{title} {summary}")
                
                if texts:
                    score, _ = self._score_texts(texts)
                    self._update_source(name, score)
                    log.debug(f"[{name}] score={score:.3f} items={len(texts)}")
            except Exception as exc:
                log.warning(f"[Sentiment/{name}] Fetch failed: {exc}")
            self._sleep(interval)

    # ── Scoring & Aggregation ─────────────────────────────────────────────────

    def _score_texts(self, texts: List[str]) -> Tuple[float, bool]:
        """
        Score a list of text strings.
        Returns (normalized_score, extreme_event_detected).
        """
        if not texts:
            return 0.5, False

        total_vader = 0.0
        total_tb    = 0.0
        extreme_event = False

        for text in texts:
            text_lower = text.lower()
            
            # 1. Extreme event detection
            if any(k in text_lower for k in config.EXTREME_EVENT_KEYWORDS):
                extreme_event = True

            # 2. Base NLP
            v_score = self.vader.polarity_scores(text)["compound"]
            t_score = TextBlob(text).sentiment.polarity

            # 3. Keyword boosting
            bull_count = sum(text_lower.count(k) for k in config.BULLISH_KEYWORDS)
            bear_count = sum(text_lower.count(k) for k in config.BEARISH_KEYWORDS)
            
            if bull_count > bear_count:
                v_score = min(1.0, v_score + 0.2)
                t_score = min(1.0, t_score + 0.2)
            elif bear_count > bull_count:
                v_score = max(-1.0, v_score - 0.2)
                t_score = max(-1.0, t_score - 0.2)

            total_vader += v_score
            total_tb    += t_score

        avg_vader = total_vader / len(texts)
        avg_tb    = total_tb / len(texts)
        
        # Blend: 70% VADER, 30% TextBlob
        compound = (avg_vader * 0.7) + (avg_tb * 0.3)
        
        # Map [-1, 1] to [0, 1]
        norm_score = (compound + 1.0) / 2.0
        return norm_score, extreme_event

    def _aggregate_loop(self, interval: int, initial_delay: int):
        time.sleep(initial_delay)
        log.debug(f"[Sentiment/Aggregate] Thread started (interval={interval}s, initial={initial_delay}s)")
        while self._running:
            try:
                self._aggregate()
            except Exception as exc:
                log.error(f"[Sentiment/Aggregate] Error: {exc}", exc_info=True)
            self._sleep(interval)

    def _aggregate(self):
        """Compute the weighted composite sentiment score."""
        with self._cache_lock:
            sources = self._state["sources"]

        w = {
            "fng":           config.SENT_W_FNG,
            "cointelegraph": config.SENT_W_COINTELEGRAPH,
            "decrypt":       config.SENT_W_DECRYPT,
            "binance_ann":   config.SENT_W_BINANCE_ANN,
            "bitcoincom":    config.SENT_W_BITCOINCOM,
            "cryptopotato":  config.SENT_W_CRYPTOPOTATO,
        }

        # Filter out stale sources (> 4 hours old)
        now = time.time()
        active_w = 0.0
        weighted_sum = 0.0
        active_count = 0

        for name, weight in w.items():
            src = sources.get(name, {})
            updated = src.get("updated", 0)
            if now - updated < 4 * 3600:
                active_w += weight
                weighted_sum += src.get("score", 0.5) * weight
                active_count += 1

        if active_w > 0:
            final_score = weighted_sum / active_w
        else:
            final_score = 0.5

        # Force penalty if extreme event detected recently
        extreme = False
        if any(s.get("extreme_event") for s in sources.values() if now - s.get("updated",0) < 3600):
            extreme = True
            final_score = max(0.0, final_score - 0.15)

        # Labeling
        if final_score >= config.SENT_LABEL_VERY_BULLISH:
            label = "VERY_BULLISH"
        elif final_score >= config.SENT_LABEL_BULLISH:
            label = "BULLISH"
        elif final_score >= config.SENT_LABEL_NEUTRAL_HIGH:
            label = "NEUTRAL"
        elif final_score >= config.SENT_LABEL_NEUTRAL_LOW:
            label = "NEUTRAL"
        elif final_score >= config.SENT_LABEL_BEARISH:
            label = "BEARISH"
        else:
            label = "VERY_BEARISH"

        with self._cache_lock:
            self._state["composite"] = {
                "score": round(final_score, 4),
                "label": label,
                "extreme_event": extreme,
                "updated": now,
            }
        
        self._save_cache()
        log.debug(f"[Sentiment] score={final_score:.3f} label={label} extreme={extreme} active={active_count}/7")

    # ── State & Caching ───────────────────────────────────────────────────────

    def _update_source(self, name: str, score: float, extreme_event: bool = False):
        with self._cache_lock:
            self._state["sources"][name] = {
                "score": score,
                "extreme_event": extreme_event,
                "updated": time.time(),
            }

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(config.SENTIMENT_CACHE_PATH), exist_ok=True)
            with open(config.SENTIMENT_CACHE_PATH, "w") as f:
                json.dump(self._state, f)
        except Exception as exc:
            log.warning(f"[Sentiment] Failed to save cache: {exc}")

    def _load_cache(self):
        try:
            if os.path.exists(config.SENTIMENT_CACHE_PATH):
                with open(config.SENTIMENT_CACHE_PATH, "r") as f:
                    data = json.load(f)
                    self._state.update(data)
                log.info(f"[Sentiment] Loaded cache: score={self._state['composite']['score']:.2f} label={self._state['composite']['label']}")
        except Exception as exc:
            log.warning(f"[Sentiment] Failed to load cache: {exc}")

    def _sleep(self, seconds: int):
        """Interruptible sleep."""
        for _ in range(seconds):
            if not self._running:
                break
            time.sleep(1)
