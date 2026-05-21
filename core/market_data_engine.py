"""
core/market_data_engine.py — OHLCV fetch engine for Binance Futures.

Responsibilities:
  - Connect to Binance USDT-M Futures (Testnet or Live) via ccxt
  - Fetch OHLCV for 5m (execution) and 15m (trend) timeframes
  - Deduplicate: only return newly closed candles
  - Return clean pandas DataFrames ready for feature engineering
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import ccxt
import pandas as pd

import config
from core.logger import setup_logger

log = setup_logger("MarketData")


# ══════════════════════════════════════════════════════════════════════════════
# Exchange Connection
# ══════════════════════════════════════════════════════════════════════════════

def _build_exchange() -> ccxt.Exchange:
    """Create and configure the ccxt Binance exchange object for public market data."""
    exchange = ccxt.binance({
        "options": {
            "defaultType": "future", # Use Binance Futures
        },
        "enableRateLimit": True,
    })
    log.info("[MarketData] Running against Binance Futures LIVE")
    return exchange


# ══════════════════════════════════════════════════════════════════════════════
# MarketDataEngine
# ══════════════════════════════════════════════════════════════════════════════

class MarketDataEngine:
    """
    Thin wrapper around ccxt that fetches OHLCV data and tracks the
    last-processed candle per timeframe to prevent duplicate processing.

    Usage:
        mde = MarketDataEngine()
        df_5m, df_15m = mde.fetch_both()     # returns (5m_df, 15m_df) or (None, None)
        # On a newly closed candle df_5m will be non-None, otherwise None.
    """

    COLS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self):
        self.exchange = _build_exchange()
        self._symbol = config.SYMBOL.replace(":USDT", "")
        # Track the timestamp (ms) of the last candle we processed per timeframe
        self._last_ts: Dict[str, int] = {
            config.TF_EXEC:  0,
            config.TF_TREND: 0,
        }
        self._retries = 3
        self._retry_delay = 2.0

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_both(self, tick_mode: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Fetch 5m and 15m OHLCV.
        Returns (df_5m, df_15m).
        If tick_mode is True, returns live/open candle data continuously.
        If tick_mode is False, df_5m is None if the latest candle is still open (not yet closed).
        """
        df_5m  = self._fetch_tf(config.TF_EXEC,  config.LOOKBACK_CANDLES_5M, tick_mode=tick_mode)
        df_15m = self._fetch_tf(config.TF_TREND, config.LOOKBACK_CANDLES_15M, tick_mode=tick_mode)

        if df_5m is None or df_15m is None:
            return None, None

        if tick_mode:
            return df_5m, df_15m

        # Check if the most recent 5m candle is newly closed
        latest_ts = int(df_5m.iloc[-1]["timestamp"])
        if latest_ts <= self._last_ts[config.TF_EXEC]:
            # Already processed this candle
            return None, df_15m

        self._last_ts[config.TF_EXEC] = latest_ts
        log.info(
            f"[MarketData] New 5m candle @ "
            f"{datetime.fromtimestamp(latest_ts/1000, tz=timezone.utc).strftime('%H:%M:%S UTC')} "
            f"close={df_5m.iloc[-1]['close']:.2f}"
        )
        return df_5m, df_15m

    def fetch_training_data(self, days: int = config.TRAINING_DAYS) -> pd.DataFrame:
        """
        Fetch the last `days` days of 5m candles for ML training.
        Stitches multiple requests together if needed.
        """
        limit_per_call = config.CANDLE_LIMIT
        total_candles  = days * 24 * 12          # 12 candles per hour at 5m
        all_ohlcv: list = []

        since_ms = int((time.time() - days * 86400) * 1000)
        fetched   = 0

        log.info(f"[MarketData] Fetching {days}d training data (~{total_candles} candles)…")

        while fetched < total_candles:
            batch = self._fetch_raw(config.TF_EXEC, limit_per_call, since_ms)
            if not batch:
                break
            all_ohlcv.extend(batch)
            since_ms = batch[-1][0] + 1          # next fetch starts after last ts
            fetched  += len(batch)
            if len(batch) < limit_per_call:
                break                             # reached the end of available data
            time.sleep(0.5)                       # rate-limit courtesy

        if not all_ohlcv:
            log.error("[MarketData] No training data fetched!")
            return pd.DataFrame()

        df = self._to_df(all_ohlcv)
        # Drop the last (potentially open) candle
        df = df.iloc[:-1].copy()
        log.info(f"[MarketData] Training data: {len(df)} candles fetched.")
        return df

    def get_balance(self) -> float:
        """Deprecated: Balance should be fetched from ExecutionEngine."""
        log.warning("[MarketData] get_balance is deprecated in MarketDataEngine. Query ExecutionEngine instead.")
        return 0.0

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _fetch_tf(self, timeframe: str, limit: int, tick_mode: bool = False) -> Optional[pd.DataFrame]:
        """Fetch OHLCV for one timeframe, with retry logic."""
        raw = self._fetch_raw(timeframe, limit)
        if raw is None:
            return None
        df = self._to_df(raw)
        if not tick_mode:
            # Drop the most-recent candle if it's still open (current live candle)
            # A closed candle's close_time = open_time + tf_ms - 1
            tf_ms = self._tf_to_ms(timeframe)
            now_ms = int(time.time() * 1000)
            close_time_ms = int(df.iloc[-1]["timestamp"]) + tf_ms
            if close_time_ms > now_ms:
                df = df.iloc[:-1].copy()             # remove open candle
        return df if not df.empty else None

    def _fetch_raw(
        self,
        timeframe: str,
        limit: int,
        since_ms: Optional[int] = None,
    ) -> Optional[list]:
        """Fetch raw OHLCV from ccxt with retries."""
        for attempt in range(1, self._retries + 1):
            try:
                params: dict = {}
                ohlcv = self.exchange.fetch_ohlcv(
                    self._symbol, timeframe,
                    since=since_ms, limit=limit, params=params
                )
                return ohlcv
            except ccxt.NetworkError as exc:
                log.warning(f"[MarketData] Network error (attempt {attempt}): {exc}")
            except ccxt.ExchangeError as exc:
                log.error(f"[MarketData] Exchange error: {exc}")
                return None
            except Exception as exc:
                log.error(f"[MarketData] Unexpected error: {exc}")
                return None
            time.sleep(self._retry_delay)
        return None

    @classmethod
    def _to_df(cls, ohlcv: list) -> pd.DataFrame:
        df = pd.DataFrame(ohlcv, columns=cls.COLS)
        df["timestamp"] = df["timestamp"].astype("int64")
        df[["open", "high", "low", "close", "volume"]] = (
            df[["open", "high", "low", "close", "volume"]].astype("float64")
        )
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    @staticmethod
    def _tf_to_ms(timeframe: str) -> int:
        """Convert ccxt timeframe string to milliseconds."""
        units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
        return int(timeframe[:-1]) * units[timeframe[-1]]
