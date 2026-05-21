"""
main.py — Aggressive Opportunity-Seeking EA Orchestrator.

Pipeline per newly closed 5m candle:
  1.  MarketDataEngine   → 5m + 15m OHLCV
  2.  FeatureEngine      → indicators + momentum features
  3.  MLEngine           → LONG_PROB / SHORT_PROB (binary)
  4.  SentimentEngine    → composite sentiment (background cache)
  5.  LLMEngine          → confidence adjustment
  6.  SignalEngine       → LONG_SCORE vs SHORT_SCORE → directional signal
  7.  RiskManager        → circuit breaker check
  8.  PositionManager    → state machine: open / monitor / reverse
  9.  TradeMonitor       → TP1/TP2/trailing/momentum exit/reversal
  10. DBLogger           → SQLite record

Run modes:
  python main.py           → live testnet
  python main.py --dry-run → signals only, no orders
  python main.py --retrain → force ML retrain then exit
  python main.py --backtest → walk-forward backtest then exit
"""

from __future__ import annotations

import argparse
import os
import signal
import ssl
import sys
import time

# Bypass SSL verification errors (common on Windows with demo/live Binance APIs)
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import config
from core.logger import setup_logger, DBLogger
from core.market_data_engine import MarketDataEngine
from core.feature_engine import FeatureEngine
from core.ml_engine import MLEngine
from core.llm_engine import LLMEngine
from core.sentiment_engine import SentimentEngine
from core.signal_engine import SignalEngine
from core.swarm_bridge import SwarmBridge
from core.risk_manager import RiskManager
from core.position_manager import PositionManager
from core.execution_engine import ExecutionEngine
from core.trade_monitor import TradeMonitor

log = setup_logger("Main")
db  = DBLogger()

for d in (config.LOG_DIR, config.DATA_DIR, config.MODEL_DIR):
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Expert Advisor
# ══════════════════════════════════════════════════════════════════════════════

class ExpertAdvisor:

    def __init__(self, dry_run: bool = False):
        self.dry_run  = dry_run
        self._running = True

        log.info("=" * 60)
        log.info("  Crypto Futures Scalping Expert Advisor  [AGGRESSIVE MODE]")
        log.info(f"  Symbol: {config.SYMBOL}  |  TF: {config.TF_EXEC}/{config.TF_TREND}")
        log.info(f"  Mode: {'DRY RUN (signals only)' if dry_run else 'LIVE TESTNET'}")
        log.info(f"  Thresholds: entry={config.CONF_NO_POSITION} existing={config.CONF_EXISTING} reversal={config.CONF_REVERSAL}")
        log.info("=" * 60)

        self.mde       = MarketDataEngine()
        self.fe        = FeatureEngine()
        self.ml        = MLEngine()
        self.llm       = LLMEngine()
        self.sentiment = SentimentEngine()
        self.swarm     = SwarmBridge()
        self.se        = SignalEngine()
        self.exec_eng  = ExecutionEngine()
        self.risk      = RiskManager()
        self.pm        = PositionManager(self.exec_eng, self.risk)
        self.monitor   = TradeMonitor()

        self.sentiment.start()

        if not dry_run:
            self.exec_eng.set_leverage()
            self.pm.sync_state_with_exchange()

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        log.info("[EA] Starting main loop…")

        if not self.ml.ensure_trained(self.mde):
            log.error("[EA] Could not train ML model. Exiting.")
            sys.exit(1)

        log.info("[EA] Model ready. Hunting opportunities…")
        last_retrain_check = time.time()

        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                log.error(f"[EA] Unhandled exception: {exc}", exc_info=True)
                time.sleep(5)

            if time.time() - last_retrain_check > 3600:
                last_retrain_check = time.time()
                if self.ml.is_retrain_due():
                    log.info("[EA] Scheduled retrain triggered.")
                    self.ml.ensure_trained(self.mde)

            time.sleep(config.MAIN_LOOP_SLEEP_S)

        self._shutdown()

    # ── Single Candle Tick ────────────────────────────────────────────────────

    def _tick(self):
        # 1. Fetch data
        df_5m, df_15m = self.mde.fetch_both()
        if df_5m is None:
            return

        # 2. Compute features
        feats = self.fe.get_latest_features(df_5m, df_15m)
        close = feats.get("close", 0.0)
        atr   = feats.get("atr",   0.0)

        # 3. ML prediction (binary: long_prob vs short_prob)
        feat_row = self.fe.build_ml_feature_row(feats)
        ml_probs = self.ml.predict(feat_row)

        # 4. Sentiment (cached — non-blocking)
        sent = self.sentiment.get_sentiment()

        # 5. LLM context
        llm_result = self.llm.analyze(feats, ml_probs, sent, position=self.pm.state)

        # 5.5 Swarm context
        current_data = {
            'close': close,
            'high': feats.get('high', close),
            'low': feats.get('low', close),
            'open': feats.get('open', close),
            'volume': feats.get('volume', 0.0)
        }
        swarm_result = self.swarm.analyze(current_data, feats)

        # 6. Generate signal — always LONG or SHORT, threshold depends on state
        has_position = self.pm.state != "NONE"
        signal_out   = self.se.evaluate(
            ml_probs, feats, sent, llm_result,
            swarm_result=swarm_result,
            has_position=has_position,
            current_side=self.pm.state,
        )

        log.info(
            f"[EA] {signal_out['action']} "
            f"long={signal_out.get('long_score', 0):.3f} "
            f"short={signal_out.get('short_score', 0):.3f} "
            f"| conf={signal_out.get('confidence', 0):.3f} "
            f"sent={sent.get('label')} "
            f"llm={llm_result.get('confidence_adjustment', 0):+.3f} "
            f"swarm={swarm_result.get('signal', 'HOLD')} ({swarm_result.get('confidence', 0):.2f})"
        )

        db.log_signal(
            symbol=config.SYMBOL,
            timeframe=config.TF_EXEC,
            action=signal_out["action"],
            confidence=signal_out.get("confidence", 0),
            ml_probs=ml_probs,
            tech_score=signal_out["breakdown"].get("tech_long", 0),
            sent_score=sent.get("score", 0.5),
            llm_adj=llm_result.get("confidence_adjustment", 0),
            regime=feats.get("regime", ""),
            trend_dir=feats.get("trend_dir", ""),
            raw=signal_out.get("breakdown"),
        )

        if self.dry_run:
            return

        balance = self.exec_eng.get_balance()
        self.risk.update_balance(balance)

        # 7. If position is open → Monitor Mode
        if has_position:
            monitor_action = self.monitor.check(
                feats, self.pm.state, signal_out,
                self.exec_eng, self.pm, self.pm.open_ts
            )
            log.debug(f"[EA] Monitor → {monitor_action}")

            if monitor_action == "REVERSE":
                # Delegate reversal to position manager with the latest signal
                result = self.pm.process_signal(signal_out, feats, balance)
                log.info(f"[EA] Reversal executed: {result}")
            # CLOSE_EARLY and UPDATE_TRAIL are handled internally by monitor/pm
            return

        # 8. No position → try to enter
        result = self.pm.process_signal(signal_out, feats, balance)

        if result.startswith("OPEN"):
            # Initialise trailing stop tracker with new position's SL
            self.monitor.reset(self.pm.sl_price)
            log.info(
                f"[EA] {result} | state={self.pm.state} "
                f"SL={self.pm.sl_price:.2f} "
                f"TP={self.pm.tp_price:.2f} "
                f"balance={balance:.2f}"
            )

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self, *args):
        log.info("[EA] Shutting down…")
        self._running = False
        self.sentiment.stop()

        if not self.dry_run and self.pm.state != "NONE":
            log.warning("[EA] Closing open position on shutdown…")
            self.pm.close_trade(reason="shutdown")

        log.info("[EA] Shutdown complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Crypto Futures Scalping EA — Aggressive Mode")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Signal generation only — no orders placed.")
    parser.add_argument("--retrain",  action="store_true",
                        help="Force ML retrain then exit.")
    parser.add_argument("--backtest", action="store_true",
                        help="Run walk-forward backtest then exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.backtest:
        log.info("[Main] Backtest mode…")
        from backtest.backtester import Backtester
        mde = MarketDataEngine()
        df  = mde.fetch_training_data(days=config.BT_TOTAL_DAYS)
        if df.empty:
            log.error("[Main] No data for backtest.")
            sys.exit(1)
        bt = Backtester(initial_balance=1000.0)
        bt.print_summary(bt.run_walk_forward(df))
        sys.exit(0)

    if args.retrain:
        log.info("[Main] Force-retraining ML model…")
        mde = MarketDataEngine()
        ml  = MLEngine()
        if os.path.exists(config.MODEL_PATH):
            os.remove(config.MODEL_PATH)
            log.info(f"[Main] Removed old model: {config.MODEL_PATH}")
        success = ml.ensure_trained(mde)
        log.info(f"[Main] Retrain {'successful' if success else 'FAILED'}.")
        sys.exit(0 if success else 1)

    ea = ExpertAdvisor(dry_run=args.dry_run)
    ea.run()
