"""
core/ml_engine.py — XGBoost Directional Probability Engine.

Architecture change:
  OLD: 3-class (LONG / SHORT / HOLD)
  NEW: 2-class (LONG=0 / SHORT=1) with sample weights to fix imbalance.
       HOLD emerges from insufficient confidence, NOT from a model class.

Label Engineering (improved):
  Lookahead = 4 candles (20 min at 5m).
  TP = ATR × 1.5,  SL = ATR × 0.5.
  LONG  → future_high reaches TP first.
  SHORT → future_low  reaches SL first.
  Ambiguous candles (both hit same candle, or neither) are dropped.

Output:
  {"long_prob": 0.78, "short_prob": 0.22}

Sample weights:
  sklearn compute_sample_weight("balanced") → compensates class imbalance.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

import config
from core.feature_engine import FeatureEngine, FEATURE_COLS
from core.logger import setup_logger

log = setup_logger("MLEngine")


# ══════════════════════════════════════════════════════════════════════════════
# Label Engineering (Binary: LONG vs SHORT only)
# ══════════════════════════════════════════════════════════════════════════════

def _build_labels(df: pd.DataFrame) -> pd.Series:
    """
    Symmetric Label Engineering:
    - LONG setup: Price goes UP by 1.5×ATR (TP) before going DOWN by 0.5×ATR (SL).
    - SHORT setup: Price goes DOWN by 1.5×ATR (TP) before going UP by 0.5×ATR (SL).
    
    If only LONG setup wins, labeled config.LABEL_LONG (0).
    If only SHORT setup wins, labeled config.LABEL_SHORT (1).
    All other candles (neutral, double SL hit, no TP hit) are left as NaN and dropped.
    """
    horizon = config.PREDICTION_HORIZON
    labels  = np.full(len(df), np.nan)

    close = df["close"].to_numpy()
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    atr   = df["atr"].to_numpy()

    tp_mult = config.TP_ATR_MULT    # 1.5×ATR
    sl_mult = config.SL_ATR_MULT     # 0.5×ATR

    for i in range(len(df) - horizon):
        # 1. Evaluate LONG Setup path
        tp_long = close[i] + atr[i] * tp_mult
        sl_long = close[i] - atr[i] * sl_mult
        long_status = None
        
        for j in range(i + 1, i + 1 + horizon):
            hit_tp = high[j] >= tp_long
            hit_sl = low[j] <= sl_long
            if hit_tp and not hit_sl:
                long_status = "WIN"
                break
            elif hit_sl and not hit_tp:
                long_status = "LOSS"
                break
            elif hit_tp and hit_sl:
                long_status = "LOSS"
                break

        # 2. Evaluate SHORT Setup path
        tp_short = close[i] - atr[i] * tp_mult
        sl_short = close[i] + atr[i] * sl_mult
        short_status = None
        
        for j in range(i + 1, i + 1 + horizon):
            hit_tp = low[j] <= tp_short
            hit_sl = high[j] >= sl_short
            if hit_tp and not hit_sl:
                short_status = "WIN"
                break
            elif hit_sl and not hit_tp:
                short_status = "LOSS"
                break
            elif hit_tp and hit_sl:
                short_status = "LOSS"
                break

        # Symmetrical assignment
        if long_status == "WIN" and short_status != "WIN":
            labels[i] = config.LABEL_LONG
        elif short_status == "WIN" and long_status != "WIN":
            labels[i] = config.LABEL_SHORT

    return pd.Series(labels, index=df.index, name="label")


# ══════════════════════════════════════════════════════════════════════════════
# MLEngine
# ══════════════════════════════════════════════════════════════════════════════

class MLEngine:
    """
    Binary XGBoost classifier: LONG (0) vs SHORT (1).

    Usage:
        ml = MLEngine()
        ml.ensure_trained(market_data_engine)
        probs = ml.predict(feature_row_df)
        # → {"long_prob": 0.73, "short_prob": 0.27}
    """

    def __init__(self):
        self._model: Optional[xgb.XGBClassifier] = None
        self._last_trained_at: float = 0.0
        self._fe = FeatureEngine()
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.DATA_DIR,  exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def ensure_trained(self, mde) -> bool:
        """Load from disk if fresh enough; otherwise retrain."""
        model_exists  = os.path.exists(config.MODEL_PATH)
        hours_since   = (time.time() - self._last_trained_at) / 3600

        if model_exists and hours_since < config.RETRAIN_INTERVAL_H:
            if self._model is None:
                self._load()
            return self._model is not None

        log.info("[ML] Starting model training…")
        synthetic_path = "data/synthetic_ohlcv.csv"
        
        if os.path.exists(synthetic_path):
            log.info(f"[ML] Found synthetic dataset → {synthetic_path}. Loading...")
            df_raw = pd.read_csv(synthetic_path)
        else:
            log.info("[ML] No synthetic dataset found. Fetching live exchange training data...")
            df_raw = mde.fetch_training_data(days=config.TRAINING_DAYS)
            
        if df_raw is None or df_raw.empty:
            log.error("[ML] Training aborted — no data.")
            return False

        success = self._train(df_raw)
        if success:
            self._save()
            self._last_trained_at = time.time()
        return success

    def predict(self, feature_row: pd.DataFrame) -> Dict[str, float]:
        """
        Predict LONG/SHORT probabilities for a single candle.
        Always returns a directional answer — confidence decides if we act.
        """
        if self._model is None:
            log.warning("[ML] Model not loaded — returning balanced probs.")
            return {"long_prob": 0.50, "short_prob": 0.50}

        proba = self._model.predict_proba(feature_row)[0]   # shape (2,)
        return {
            "long_prob":  float(proba[config.LABEL_LONG]),    # class 0
            "short_prob": float(proba[config.LABEL_SHORT]),   # class 1
        }

    def is_retrain_due(self) -> bool:
        elapsed_h = (time.time() - self._last_trained_at) / 3600
        return elapsed_h >= config.RETRAIN_INTERVAL_H

    # ── Training ──────────────────────────────────────────────────────────────

    def _train(self, df_raw: pd.DataFrame) -> bool:
        """Full training pipeline: features → labels → XGBoost fit."""
        try:
            df = self._fe.compute(df_raw)
            df = df.dropna(subset=FEATURE_COLS).copy()

            labels = _build_labels(df)
            df["label"] = labels
            # Drop ambiguous (NaN) candles and last PREDICTION_HORIZON rows
            df = df.dropna(subset=["label"]).copy()
            df = df.iloc[:-config.PREDICTION_HORIZON].copy()

            X = df[FEATURE_COLS].values
            y = df["label"].values.astype(int)

            if len(X) < 200:
                log.error(f"[ML] Too few samples: {len(X)}. Skipping train.")
                return False

            n_long  = int((y == config.LABEL_LONG).sum())
            n_short = int((y == config.LABEL_SHORT).sum())
            log.info(f"[ML] Dataset: {len(X)} samples | LONG={n_long} SHORT={n_short}")

            # Balanced sample weights (fix class imbalance)
            sample_weights = compute_sample_weight("balanced", y)

            X_train, X_val, y_train, y_val, sw_train, _ = train_test_split(
                X, y, sample_weights,
                test_size=0.15, shuffle=False
            )

            model = xgb.XGBClassifier(
                **{k: v for k, v in config.XGB_PARAMS.items()
                   if k != "use_label_encoder"},
                objective="binary:logistic",
            )
            model.fit(
                X_train, y_train,
                sample_weight=sw_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Validation report
            y_pred = model.predict(X_val)
            report = classification_report(
                y_val, y_pred,
                target_names=["LONG", "SHORT"],
                zero_division=0,
            )
            log.info(f"[ML] Validation report:\n{report}")

            # Feature importance top-5
            importances = model.feature_importances_
            top5 = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])[:5]
            log.info("[ML] Top-5 features: " +
                     ", ".join(f"{n}={v:.3f}" for n, v in top5))

            self._model = model
            return True

        except Exception as exc:
            log.error(f"[ML] Training failed: {exc}", exc_info=True)
            return False

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        try:
            self._model.save_model(config.MODEL_PATH)
            log.info(f"[ML] Model saved → {config.MODEL_PATH}")
        except Exception as exc:
            log.error(f"[ML] Save failed: {exc}")

    def _load(self):
        try:
            model = xgb.XGBClassifier(
                **{k: v for k, v in config.XGB_PARAMS.items()
                   if k != "use_label_encoder"},
                objective="binary:logistic",
            )
            model.load_model(config.MODEL_PATH)
            self._model = model
            mtime = os.path.getmtime(config.MODEL_PATH)
            self._last_trained_at = mtime
            log.info(
                f"[ML] Model loaded from {config.MODEL_PATH} "
                f"(trained {(time.time()-mtime)/3600:.1f}h ago)"
            )
        except Exception as exc:
            log.warning(f"[ML] Load failed: {exc}. Will retrain.")
            self._model = None
