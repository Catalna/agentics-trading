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
from core.feature_engine import FeatureEngine, FEATURE_COLS
from core.ml_engine import MLEngine, _build_labels
from core.signal_engine import SignalEngine
from core.logger import setup_logger

log = setup_logger("Backtester")


# ══════════════════════════════════════════════════════════════════════════════
# Cost Model
# ══════════════════════════════════════════════════════════════════════════════

def _apply_costs(pnl: float, notional: float, hold_candles: int) -> float:
    """Subtract trading costs from gross P&L."""
    fee_open   = notional * config.BT_TAKER_FEE
    fee_close  = notional * config.BT_TAKER_FEE
    slippage   = notional * config.BT_SLIPPAGE
    # Funding: every 8h = every 96 candles at 5m
    funding_periods = hold_candles / 96
    funding  = notional * config.BT_FUNDING_FEE * funding_periods
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

    # ── Walk-Forward ──────────────────────────────────────────────────────────

    def run_walk_forward(self, df_raw: pd.DataFrame) -> List[Dict]:
        """
        Run walk-forward validation over df_raw.
        Returns list of per-window result dicts.
        """
        # Compute features once
        df = self._fe.compute(df_raw).dropna(subset=FEATURE_COLS).copy()
        df = df.reset_index(drop=True)

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
        """Train the ML model on a training window (in-memory, no file I/O)."""
        from sklearn.model_selection import train_test_split
        import xgboost as xgb

        try:
            labels = _build_labels(train_df)
            train_df = train_df.copy()
            train_df["label"] = labels
            train_df = train_df.dropna(subset=["label"]).iloc[: -config.PREDICTION_HORIZON]

            X = train_df[FEATURE_COLS].values
            y = train_df["label"].values.astype(int)

            if len(X) < 200:
                log.warning(f"[BT] Too few training samples: {len(X)}")
                return False

            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)

            model = xgb.XGBClassifier(
                **{k: v for k, v in config.XGB_PARAMS.items()
                   if k != "use_label_encoder"},
                objective="binary:logistic",
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            ml._model = model
            return True
        except Exception as exc:
            log.error(f"[BT] Train failed: {exc}")
            return False

    # ── Simulation ────────────────────────────────────────────────────────────

    def _simulate(self, ml: MLEngine, df: pd.DataFrame) -> Dict:
        """Simulate trading on test window. Returns metrics dict."""
        balance      = self._initial_balance
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
            feats = {col: row[col] for col in FEATURE_COLS}
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
                "trend_15m":"NEUTRAL",
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
                    notional = entry_price * (self._position_qty(balance, atr, entry_price))
                    gross_pnl = (
                        (exit_price - entry_price) if state == "LONG" else (entry_price - exit_price)
                    ) * self._position_qty(balance, atr, entry_price) * config.LEVERAGE
                    net_pnl = _apply_costs(gross_pnl, notional, candles_held)

                    balance  = max(0.0, balance + net_pnl)
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
                feat_row = self._fe.build_ml_feature_row(feats)
                ml_probs = ml.predict(feat_row)
                signal   = self._se.evaluate(ml_probs, feats, neutral_sentiment, neutral_llm)

                if signal["action"] in ("LONG", "SHORT") and balance > 0:
                    side       = signal["action"]
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
            candles_held = len(df) - entry_idx
            notional = entry_price * self._position_qty(balance, df.iloc[-1]["atr"], entry_price)
            gross_pnl = (
                (final_price - entry_price) if state == "LONG" else (entry_price - final_price)
            ) * self._position_qty(balance, df.iloc[-1]["atr"], entry_price) * config.LEVERAGE
            net_pnl = _apply_costs(gross_pnl, notional, candles_held)
            balance = max(0.0, balance + net_pnl)
            trades.append({
                "side": state, "entry": entry_price, "exit": final_price,
                "pnl": net_pnl, "candles": candles_held, "reason": "EOW",
            })

        return self._compute_metrics(trades, equity_curve, balance)

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def _position_qty(balance: float, atr: float, price: float) -> float:
        """Replicate risk manager sizing for backtest (mid-tier 1% risk)."""
        risk   = balance * config.RISK_TIER_MID
        sl_usd = atr * config.SL_ATR_MULT
        qty    = risk / sl_usd if sl_usd > 0 else 0.001
        return max(qty, config.MIN_POSITION_NOTIONAL / price)

    @staticmethod
    def _compute_metrics(trades: List[Dict], equity: List[float], final_balance: float) -> Dict:
        if not trades:
            return {
                "total_trades": 0, "win_rate": 0, "profit_factor": 0,
                "sharpe": 0, "max_drawdown_pct": 0,
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
    import time as _time

    log.info(f"[BT] Starting {config.BT_TOTAL_DAYS}-day walk-forward backtest for {config.SYMBOL}…")
    log.info("[BT] Fetching historical data…")

    from core.market_data_engine import MarketDataEngine
    mde = MarketDataEngine()
    df_raw = mde.fetch_training_data(days=config.BT_TOTAL_DAYS)

    if df_raw.empty:
        log.error("[BT] No data fetched. Exiting.")
        sys.exit(1)

    log.info(f"[BT] {len(df_raw)} candles fetched. Running walk-forward…")
    bt      = Backtester(initial_balance=1000.0)
    results = bt.run_walk_forward(df_raw)
    bt.print_summary(results)
