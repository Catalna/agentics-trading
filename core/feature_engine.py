"""
core/feature_engine.py — Technical Indicator Computation & Normalization.

Computes:
  EMA5, EMA20, RSI(14), MACD(12,26,9), ATR(14), ADX(14), VWAP,
  Bollinger Bands(20,2), Volume ratio

Normalized ML features:
  ema5_dist         = (close - ema5)  / close
  ema20_dist        = (close - ema20) / close
  vwap_dist         = (close - vwap)  / close
  atr_pct           = atr / close
  volume_change_pct = current_vol / avg_vol_20 - 1
  rsi_scaled        = rsi / 100
  adx_scaled        = adx / 100
  macd_norm         = macd_hist / close
  rsi_slope         = (rsi - rsi_3_bars_ago) / 3      ← momentum
  macd_slope        = (macd_hist - macd_hist_3_ago) / close  ← momentum
  bb_dist           = (close - bb_mid) / (bb_upper - bb_lower + 1e-9)

Regime + trend detection:
  regime    : TRENDING | RANGING
  trend_dir : BULLISH  | BEARISH | NEUTRAL
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config
from core.logger import setup_logger

log = setup_logger("FeatureEngine")

# ─── Feature column names (exact list fed to XGBoost) ─────────────────────────
FEATURE_COLS = [
    "ema5_dist",
    "ema20_dist",
    "vwap_dist",
    "atr_pct",
    "volume_change_pct",
    "rsi_scaled",
    "adx_scaled",
    "macd_norm",
    "rsi_slope",       # NEW — RSI momentum
    "macd_slope",      # NEW — MACD histogram momentum
    "bb_dist",         # NEW — Bollinger Band position
]


# ══════════════════════════════════════════════════════════════════════════════
# Low-level indicator helpers (vectorised, no pandas_ta dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_l  = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast    = _ema(close, fast)
    ema_slow    = _ema(close, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed ADX."""
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    plus_dm  = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    mask_both = (plus_dm > 0) & (minus_dm > 0)
    plus_dm[mask_both & (minus_dm >= plus_dm)]  = 0
    minus_dm[mask_both & (plus_dm  > minus_dm)] = 0

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr14      = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di14  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di14 = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr14.replace(0, np.nan)

    dx  = (100 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP — cumulative within the day partition."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol    = (typical_price * df["volume"]).cumsum()
    cum_vol       = df["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def _bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0):
    """Return (upper, mid, lower) Bollinger Bands."""
    mid   = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    return upper, mid, lower


# ══════════════════════════════════════════════════════════════════════════════
# FeatureEngine
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngine:
    """
    Computes and normalises all technical features from raw OHLCV DataFrames.

    Usage:
        fe = FeatureEngine()
        df_feat = fe.compute(df_5m)                        # full DataFrame
        latest  = fe.get_latest_features(df_5m, df_15m)   # single-row dict
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all indicator and normalised feature columns to the DataFrame.
        Modifies a copy; original is untouched.
        """
        df    = df.copy()
        close = df["close"]
        high  = df["high"]
        low   = df["low"]

        # ── Raw indicators ────────────────────────────────────────────────────
        df["ema5"]    = _ema(close, config.EMA_FAST)
        df["ema20"]   = _ema(close, config.EMA_SLOW)
        df["rsi"]     = _rsi(close, config.RSI_PERIOD)

        macd_line, signal_line, macd_hist = _macd(
            close, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL
        )
        df["macd"]        = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"]   = macd_hist
        df["atr"]         = _atr(high, low, close, config.ATR_PERIOD)
        df["adx"]         = _adx(high, low, close, config.ADX_PERIOD)
        df["vwap"]        = _vwap(df)

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = _bollinger_bands(close, config.BB_PERIOD, config.BB_STD)
        df["bb_upper"] = bb_upper
        df["bb_mid"]   = bb_mid
        df["bb_lower"] = bb_lower

        # Volume ratio (current / rolling-20 avg)
        df["vol_avg20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg20"].replace(0, np.nan)

        # ── Momentum features (3-bar slope) ───────────────────────────────────
        df["rsi_slope"]  = df["rsi"].diff(3) / 3
        df["macd_slope"] = df["macd_hist"].diff(3) / close.replace(0, np.nan)

        # ── Normalised ML features ────────────────────────────────────────────
        bb_width = (bb_upper - bb_lower).replace(0, np.nan)
        df["ema5_dist"]         = (close - df["ema5"])  / close
        df["ema20_dist"]        = (close - df["ema20"]) / close
        df["vwap_dist"]         = (close - df["vwap"])  / close.replace(0, np.nan)
        df["atr_pct"]           = df["atr"] / close
        df["volume_change_pct"] = df["volume_ratio"] - 1.0   # normalised against avg
        df["rsi_scaled"]        = df["rsi"] / 100.0
        df["adx_scaled"]        = df["adx"] / 100.0
        df["macd_norm"]         = df["macd_hist"] / close
        df["bb_dist"]           = (close - bb_mid) / bb_width

        # Clip extreme outliers
        for col in ["volume_change_pct", "macd_norm", "macd_slope", "bb_dist"]:
            df[col] = df[col].clip(-5, 5)

        # ── Regime & trend ────────────────────────────────────────────────────
        df["regime"]    = np.where(df["adx"] > config.ADX_TREND_THRESHOLD, "TRENDING", "RANGING")
        df["trend_dir"] = np.where(
            df["ema5"] > df["ema20"], "BULLISH",
            np.where(df["ema5"] < df["ema20"], "BEARISH", "NEUTRAL")
        )

        return df

    def get_latest_features(
        self,
        df_5m:  pd.DataFrame,
        df_15m: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Return the feature dict for the latest closed candle.
        Optionally enriches with 15m trend context.
        """
        df  = self.compute(df_5m)
        row = df.iloc[-1]

        feats: Dict = {
            # Raw price (for SL/TP calculation — NOT fed to ML)
            "close":   float(row["close"]),
            "high":    float(row["high"]),
            "low":     float(row["low"]),
            "atr":     float(row["atr"]),
            "vwap":    float(row["vwap"]),
            "ema5":    float(row["ema5"]),
            "ema20":   float(row["ema20"]),
            "rsi":     float(row["rsi"]),
            "adx":     float(row["adx"]),
            "macd":    float(row["macd"]),
            "macd_signal": float(row["macd_signal"]),
            "macd_hist":   float(row["macd_hist"]),
            "volume_ratio": float(row["volume_ratio"]) if not pd.isna(row["volume_ratio"]) else 1.0,
            # Normalised ML features
            "ema5_dist":         float(row["ema5_dist"]),
            "ema20_dist":        float(row["ema20_dist"]),
            "vwap_dist":         float(row["vwap_dist"]),
            "atr_pct":           float(row["atr_pct"]),
            "volume_change_pct": float(row["volume_change_pct"]),
            "rsi_scaled":        float(row["rsi_scaled"]),
            "adx_scaled":        float(row["adx_scaled"]),
            "macd_norm":         float(row["macd_norm"]),
            "rsi_slope":         float(row["rsi_slope"]) if not pd.isna(row["rsi_slope"]) else 0.0,
            "macd_slope":        float(row["macd_slope"]) if not pd.isna(row["macd_slope"]) else 0.0,
            "bb_dist":           float(row["bb_dist"]) if not pd.isna(row["bb_dist"]) else 0.0,
            # Context
            "regime":    str(row["regime"]),
            "trend_dir": str(row["trend_dir"]),
            "timestamp": int(row["timestamp"]),
        }

        # Enrich with 15m trend if available
        if df_15m is not None and not df_15m.empty:
            df15 = self.compute(df_15m)
            r15  = df15.iloc[-1]
            feats["trend_15m"]  = str(r15["trend_dir"])
            feats["adx_15m"]    = float(r15["adx"])
            feats["regime_15m"] = str(r15["regime"])
            feats["ema5_15m"]   = float(r15["ema5"])
            feats["ema20_15m"]  = float(r15["ema20"])

        return feats

    def build_ml_feature_row(self, feats: Dict) -> pd.DataFrame:
        """
        Extract only the FEATURE_COLS into a single-row DataFrame ready for XGBoost.
        """
        row = {col: feats.get(col, 0.0) for col in FEATURE_COLS}
        df  = pd.DataFrame([row])[FEATURE_COLS]
        return df.fillna(0.0)
