"""
backtest/backtester.py — Walk-Forward Backtesting Engine.

Simulates the full EA pipeline on historical OHLCV data:
  Features → ML → Signal → Risk → Position → P&L

Costs applied:
  - Taker fee:    0.04% per trade (open + close)
  - Slippage:     0.05% on entry
  - Funding fee:  0.01% every 8 hours while position is open

Walk-forward validation:
  - Train window: 30 days
  - Test window:  7 days
  - Rolls forward by TEST_DAYS each iteration

Metrics reported:
  Win Rate, Profit Factor, Sharpe Ratio,
  Max Drawdown, Total Trades, Total PnL, Avg Duration
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow running as __main__ directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from core.feature_engine import FeatureEngine
from core.ml_engine import MLEngine
from core.signal_engine import SignalEngine
from core.orchestrator_engine import OrchestratorEngine
from core.risk_manager import RiskManager
from core.logger import setup_logger

log = setup_logger("Backtester")


# ══════════════════════════════════════════════════════════════════════════════
# Cost Model
# ══════════════════════════════════════════════════════════════════════════════

def _apply_costs(pnl: float, notional: float, hold_candles: int) -> float:
    """Subtract trading costs from gross P&L."""
    fee_open   = notional * getattr(config, "BT_MAKER_FEE", 0.0002)
    fee_close  = notional * getattr(config, "BT_MAKER_FEE", 0.0002)
    slippage   = notional * getattr(config, "BT_SLIPPAGE", 0.0)
    # Funding: every 8h = every 96 candles at 5m
    funding_periods = hold_candles / 96
    funding  = notional * getattr(config, "BT_FUNDING_FEE", 0.0001) * funding_periods
    return pnl - fee_open - fee_close - slippage - funding


# ══════════════════════════════════════════════════════════════════════════════
# Simulation
# ══════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Simulates the EA on a prepared feature DataFrame.

    Usage:
        bt = Backtester()
        results = bt.run_walk_forward(df_full)
        bt.print_summary(results)
    """

    def __init__(self, initial_balance: float = 1000.0):
        self._initial_balance = initial_balance
        self._fe = FeatureEngine()
        self._se = SignalEngine()
        self._llme = OrchestratorEngine()
        self._rm = RiskManager(initial_balance)

    def _add_multi_tf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Injects 15m trend and 1d macro features to mimic live environment."""
        if "timestamp" not in df.columns:
            return df
        
        # Temporary datetime index
        df["dt"] = pd.to_datetime(df["timestamp"], unit='ms')
        df = df.set_index("dt")
        
        # 15m Trend
        df_15m = df.resample('15min').agg({'close': 'last'}).dropna()
        df_15m["ema5"] = df_15m["close"].ewm(span=config.EMA_FAST, adjust=False).mean()
        df_15m["ema20"] = df_15m["close"].ewm(span=config.EMA_SLOW, adjust=False).mean()
        df_15m["trend_15m"] = np.where(df_15m["ema5"] > df_15m["ema20"], "BULLISH", "BEARISH")
        
        # 1d Macro
        df_1d = df.resample('1D').agg({'high': 'max', 'low': 'min'}).dropna()
        df_1d["macro_high"] = df_1d["high"].rolling(config.LOOKBACK_CANDLES_1D, min_periods=1).max()
        df_1d["macro_low"] = df_1d["low"].rolling(config.LOOKBACK_CANDLES_1D, min_periods=1).min()
        
        # Forward fill to 5m
        df["trend_15m"] = df_15m["trend_15m"].reindex(df.index, method='ffill')
        df["macro_high"] = df_1d["macro_high"].reindex(df.index, method='ffill')
        df["macro_low"] = df_1d["macro_low"].reindex(df.index, method='ffill')
        
        return df.reset_index(drop=False)

    # ── Walk-Forward ──────────────────────────────────────────────────────────

    def run_walk_forward(self, df_raw: pd.DataFrame) -> List[Dict]:
        """
        Run walk-forward validation over df_raw.
        Returns list of per-window result dicts.
        """
        # Compute features once
        df = self._fe.compute(df_raw).copy()
        df = self._add_multi_tf_features(df)
        
        # 4. Drop NaNs
        feature_cols = self._fe.get_feature_columns(df)
        df = df.dropna(subset=feature_cols).copy()
        
        # We need ML features in the final dataframe for ML training
        log.info(f"[BT] Processed dataset shape: {df.shape}")
        
        train_candles = config.BT_TRAIN_DAYS * 24 * 12
        test_candles  = config.BT_TEST_DAYS  * 24 * 12
        window_size   = train_candles + test_candles

        if len(df) < window_size:
            log.error(f"[BT] Not enough data ({len(df)} candles, need {window_size}).")
            return []

        windows = []
        start   = 0
        wf_num  = 1

        while start + window_size <= len(df):
            train_df = df.iloc[start : start + train_candles].copy()
            test_df  = df.iloc[start + train_candles : start + window_size].copy()

            log.info(
                f"[BT] Walk-forward #{wf_num}: "
                f"train={len(train_df)} test={len(test_df)} candles"
            )

            # Train ML on train window
            ml = MLEngine()
            success = self._train_on_window(ml, train_df)
            if not success:
                start += test_candles
                wf_num += 1
                continue

            # Simulate on test window
            result = self._simulate(ml, test_df)
            result["window"] = wf_num
            windows.append(result)
            self.print_window(result)

            start  += test_candles
            wf_num += 1

        return windows

    def _train_on_window(self, ml: MLEngine, train_df: pd.DataFrame) -> bool:
        """Train the ML model ensemble on a training window (in-memory)."""
        from sklearn.model_selection import train_test_split
        from sklearn.utils.class_weight import compute_sample_weight
        import xgboost as xgb

        try:
            # Labels are already in train_df thanks to FeatureEngine
            train_df = train_df.dropna(subset=["label"]).iloc[: -config.PREDICTION_HORIZON]

            feature_cols = self._fe.get_feature_columns(train_df)
            X = train_df[feature_cols].values
            y = train_df["label"].values.astype(int)

            if len(X) < 200:
                log.warning(f"[BT] Too few training samples: {len(X)}")
                return False

            sample_weights = compute_sample_weight("balanced", y)

            X_tr, X_val, y_tr, y_val, sw_tr, _ = train_test_split(
                X, y, sample_weights,
                test_size=0.15, shuffle=False
            )

            # MODEL 1: Conservative
            model_1 = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                max_depth=4,
                learning_rate=0.05,
                n_estimators=300,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=2.0,
                eval_metric="mlogloss",
                random_state=42,
                tree_method="hist",
                device="cuda"
            )
            
            # MODEL 2: Balanced
            model_2 = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                eval_metric="mlogloss",
                random_state=123,
                tree_method="hist",
                device="cuda"
            )
            
            # MODEL 3: Aggressive
            model_3 = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                max_depth=8,
                learning_rate=0.15,
                n_estimators=150,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=0.5,
                eval_metric="mlogloss",
                random_state=456,
                tree_method="hist",
                device="cuda"
            )

            model_1.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_val, y_val)], verbose=False)
            model_2.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_val, y_val)], verbose=False)
            model_3.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_val, y_val)], verbose=False)

            ml.models = [model_1, model_2, model_3]
            return True
        except Exception as exc:
            log.error(f"[BT] Train failed: {exc}")
            return False

    # ── Simulation ────────────────────────────────────────────────────────────

    def _simulate(self, ml: MLEngine, df: pd.DataFrame) -> Dict:
        """Simulate trading on test window. Returns metrics dict."""
        balance      = self._initial_balance
        self._rm.update_balance(balance)
        peak_balance = balance
        state        = "NONE"         # NONE | LONG | SHORT
        entry_price  = 0.0
        entry_idx    = 0
        side         = ""
        sl_price     = 0.0
        tp_price     = 0.0
        current_sl   = 0.0

        trades:       List[Dict] = []
        equity_curve: List[float] = [balance]

        # Neutral sentiment / LLM for backtest (no live feeds)
        neutral_sentiment = {"score": 0.5, "label": "NEUTRAL", "extreme_event": False}
        neutral_llm       = {"market_state": "Backtest", "confidence_adjustment": 0.0}

        for i, row in df.iterrows():
            feature_cols = self._fe.get_feature_columns(df)
            feats = {col: row[col] for col in feature_cols}
            feats.update({
                "close":    row["close"],
                "high":     row["high"],
                "low":      row["low"],
                "atr":      row["atr"],
                "rsi":      row["rsi"],
                "adx":      row["adx"],
                "macd_hist":row["macd_hist"],
                "ema5_dist":row["ema5_dist"],
                "ema20_dist":row["ema20_dist"],
                "vwap_dist":row["vwap_dist"],
                "volume_change_pct": row["volume_change_pct"],
                "volume_ratio": row.get("volume_ratio", 1.0),
                "atr_pct": row.get("atr_pct", 0.0),
                "rsi_slope": row.get("rsi_slope", 0.0),
                "macd_slope": row.get("macd_slope", 0.0),
                "regime":   row["regime"],
                "trend_dir":row["trend_dir"],
                "trend_15m":row.get("trend_15m", "NEUTRAL"),
                "macro_high":row.get("macro_high", row["high"]),
                "macro_low":row.get("macro_low", row["low"]),
            })

            close = row["close"]
            high  = row["high"]
            low   = row["low"]
            atr   = row["atr"]

            # ── Check SL/TP hit for open position ───────────────────────────
            if state != "NONE":
                hit_sl = (state == "LONG"  and low  <= sl_price) or \
                         (state == "SHORT" and high >= sl_price)
                hit_tp = (state == "LONG"  and high >= tp_price) or \
                         (state == "SHORT" and low  <= tp_price)

                if hit_sl or hit_tp:
                    exit_price = tp_price if hit_tp else sl_price
                    candles_held = int(i) - entry_idx
                    # Gunakan RiskManager yang sudah menghitung sizing, kita simpan nilainya ke self._last_qty saat open
                    qty = getattr(self, "_last_qty", 0.0)
                    notional = entry_price * qty
                    gross_pnl = (
                        (exit_price - entry_price) if state == "LONG" else (entry_price - exit_price)
                    ) * qty
                    net_pnl = _apply_costs(gross_pnl, notional, candles_held)

                    balance  = max(0.0, balance + net_pnl)
                    self._rm.update_balance(balance)
                    peak_balance = max(peak_balance, balance)

                    trades.append({
                        "side":     state,
                        "entry":    entry_price,
                        "exit":     exit_price,
                        "pnl":      net_pnl,
                        "candles":  candles_held,
                        "reason":   "TP" if hit_tp else "SL",
                    })
                    state = "NONE"
                    equity_curve.append(balance)
                    continue

                # Trailing stop update
                trail_sl = close - atr * config.TRAILING_ATR_MULT if state == "LONG" \
                           else close + atr * config.TRAILING_ATR_MULT
                if state == "LONG"  and trail_sl > current_sl:
                    current_sl = trail_sl
                    sl_price   = round(current_sl, 2)
                elif state == "SHORT" and trail_sl < current_sl:
                    current_sl = trail_sl
                    sl_price   = round(current_sl, 2)

            # ── Generate signal ─────────────────────────────────────────────
            if state == "NONE":
                feat_row = pd.DataFrame([{col: feats.get(col, 0.0) for col in feature_cols}])
                ml_probs = ml.predict(feat_row)
                signal   = self._se.evaluate(ml_probs, feats, neutral_sentiment, neutral_llm)

                if signal["action"] in ("LONG", "SHORT") and balance > 0:
                    # LLM confirmation — skipped in backtest by default for speed
                    skip_llm = getattr(config, "BACKTEST_SKIP_LLM", True)
                    if config.MULTI_LLM_ENABLED and not skip_llm:
                        log.info(f"[BT] Setup found! Querying LLM at {df.loc[i, 'dt']}...")
                        llm_result = self._llme.analyze(feats, ml_probs, neutral_sentiment, state)
                        signal = self._se.evaluate(ml_probs, feats, neutral_sentiment, llm_result)

                    if signal["action"] in ("LONG", "SHORT"):
                        # Get exact position size from RiskManager
                        qty = self._rm.get_position_size(balance, signal["confidence"], atr, close)
                        
                        if qty > 0:
                            side       = signal["action"]
                            self._last_qty = qty
                        else:
                            side       = "" # Ditolak oleh guardrail RiskManager
                    
                    if side:
                        sl_off     = atr * config.SL_ATR_MULT
                        tp_off     = atr * config.TP_ATR_MULT
                        if side == "LONG":
                            entry_price = close * (1 + config.BT_SLIPPAGE)
                            sl_price    = round(entry_price - sl_off, 2)
                            tp_price    = round(entry_price + tp_off, 2)
                        else:
                            entry_price = close * (1 - config.BT_SLIPPAGE)
                            sl_price    = round(entry_price + sl_off, 2)
                            tp_price    = round(entry_price - tp_off, 2)
                        current_sl  = sl_price
                        entry_idx   = int(i)
                        state       = side

            equity_curve.append(balance)

        # Force-close any open position at end of window
        if state != "NONE" and len(df) > 0:
            final_price = df.iloc[-1]["close"]
            candles_held = int(df.index[-1]) - entry_idx
            qty = getattr(self, "_last_qty", 0.0)
            notional = entry_price * qty
            gross_pnl = (
                (final_price - entry_price) if state == "LONG" else (entry_price - final_price)
            ) * qty
            net_pnl = _apply_costs(gross_pnl, notional, candles_held)
            balance = max(0.0, balance + net_pnl)
            self._rm.update_balance(balance)
            trades.append({
                "side": state, "entry": entry_price, "exit": final_price,
                "pnl": net_pnl, "candles": candles_held, "reason": "EOW",
            })

        return self._compute_metrics(trades, equity_curve, balance)

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def _position_qty(balance: float, atr: float, price: float) -> float:
        # NOTE: Deprecated, now handled by RiskManager instance in _simulate.
        pass

    @staticmethod
    def _compute_metrics(trades: List[Dict], equity: List[float], final_balance: float) -> Dict:
        if not trades:
            return {
                "total_trades": 0, "win_rate": 0, "profit_factor": 0,
                "sharpe_ratio": 0, "max_drawdown_pct": 0,
                "total_pnl": 0, "avg_duration_min": 0, "final_balance": final_balance,
            }

        pnls      = [t["pnl"] for t in trades]
        wins      = [p for p in pnls if p > 0]
        losses    = [p for p in pnls if p <= 0]

        win_rate  = len(wins) / len(pnls)
        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe (using per-trade returns, annualised approximately)
        returns = np.array(pnls)
        sharpe  = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

        # Max drawdown
        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        dd   = (peak - eq) / peak.clip(min=1e-9)
        max_dd = float(dd.max())

        avg_dur = np.mean([t["candles"] * 5 for t in trades])  # minutes

        return {
            "total_trades":     len(trades),
            "winning_trades":   len(wins),
            "win_rate":         round(win_rate * 100, 2),
            "profit_factor":    round(profit_factor, 3),
            "sharpe_ratio":     round(float(sharpe), 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "total_pnl":        round(sum(pnls), 4),
            "avg_duration_min": round(float(avg_dur), 1),
            "final_balance":    round(final_balance, 2),
        }

    # ── Reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def print_window(result: Dict):
        log.info(
            f"[BT] Window #{result.get('window', '?')} | "
            f"Trades={result['total_trades']} "
            f"WR={result['win_rate']:.1f}% "
            f"PF={result['profit_factor']:.2f} "
            f"Sharpe={result['sharpe_ratio']:.2f} "
            f"DD={result['max_drawdown_pct']:.1f}% "
            f"PnL={result['total_pnl']:+.2f} "
            f"AvgDur={result['avg_duration_min']:.0f}m"
        )

    def print_summary(self, results: List[Dict]):
        if not results:
            log.info("[BT] No results to summarise.")
            return

        log.info("\n" + "=" * 65)
        log.info("WALK-FORWARD BACKTEST SUMMARY")
        log.info("=" * 65)

        total_trades = sum(r["total_trades"] for r in results)
        win_rates    = [r["win_rate"] for r in results if r["total_trades"] > 0]
        pfs          = [r["profit_factor"] for r in results if r["profit_factor"] < 1e6]
        sharpes      = [r["sharpe_ratio"] for r in results]
        dds          = [r["max_drawdown_pct"] for r in results]
        pnls         = [r["total_pnl"] for r in results]
        durs         = [r["avg_duration_min"] for r in results]

        def _avg(lst): return round(sum(lst) / len(lst), 3) if lst else 0

        log.info(f"  Windows tested:     {len(results)}")
        log.info(f"  Total trades:       {total_trades}")
        log.info(f"  Avg Win Rate:       {_avg(win_rates):.1f}%  (target 55–65%)")
        log.info(f"  Avg Profit Factor:  {_avg(pfs):.3f}  (target > 1.5)")
        log.info(f"  Avg Sharpe Ratio:   {_avg(sharpes):.3f}")
        log.info(f"  Max Drawdown (max): {max(dds):.1f}%  (limit 10%)")
        log.info(f"  Total PnL:          {sum(pnls):+.2f} USDT")
        log.info(f"  Avg Trade Duration: {_avg(durs):.0f} min")
        log.info("=" * 65)

        pf_ok  = _avg(pfs) >= 1.5
        wr_ok  = 55 <= _avg(win_rates) <= 65
        dd_ok  = max(dds) < 10

        log.info(f"  PF target  ({'PASS' if pf_ok  else 'FAIL'}): {_avg(pfs):.3f}")
        log.info(f"  WR target  ({'PASS' if wr_ok  else 'FAIL'}): {_avg(win_rates):.1f}%")
        log.info(f"  DD target  ({'PASS' if dd_ok  else 'FAIL'}): {max(dds):.1f}%")
        log.info("=" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pandas as pd

    SYNTHETIC_PATH = "data/synthetic_ohlcv.csv"

    log.info(f"[BT] Starting {config.BT_TOTAL_DAYS}-day walk-forward backtest for {config.SYMBOL}…")

    # Load data: synthetic CSV first, live Binance as fallback
    df_raw = pd.DataFrame()
    if os.path.exists(SYNTHETIC_PATH):
        log.info(f"[BT] Loading synthetic dataset → {SYNTHETIC_PATH}")
        df_raw = pd.read_csv(SYNTHETIC_PATH)
        log.info(f"[BT] {len(df_raw):,} candles loaded from synthetic data.")
    else:
        log.info("[BT] No synthetic data found — fetching from Binance…")
        from core.market_data_engine import MarketDataEngine
        mde = MarketDataEngine()
        df_raw = mde.fetch_training_data(days=config.BT_TOTAL_DAYS)

    if df_raw.empty:
        log.error("[BT] No data available. Exiting.")
        sys.exit(1)

    log.info(f"[BT] Running walk-forward simulation on {len(df_raw):,} candles…")
    bt      = Backtester(initial_balance=1000.0)
    results = bt.run_walk_forward(df_raw)
    bt.print_summary(results)

