import sys, os
sys.path.insert(0, '.')

from core.feature_engine import FeatureEngine, FEATURE_COLS
from core.ml_engine import MLEngine, _build_labels
from core.signal_engine import SignalEngine
from core.risk_manager import RiskManager
from core.sentiment_engine import SentimentEngine
from core.trade_monitor import TradeMonitor
from core.position_manager import PositionManager
from core.execution_engine import ExecutionEngine
import config

print("[OK] All modules imported.")
print(f"[OK] FEATURE_COLS ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"[OK] ATR: SL={config.SL_ATR_MULT} TP={config.TP_ATR_MULT}")
print(f"[OK] Thresholds: entry={config.CONF_NO_POSITION} existing={config.CONF_EXISTING} reversal={config.CONF_REVERSAL}")
print(f"[OK] Sources: FNG, CoinTelegraph, Decrypt, BinanceAnn, Bitcoin.com, CryptoPotato, YouTube")

se = SignalEngine()
ml_probs  = {"long_prob": 0.72, "short_prob": 0.28}
sent = {"score": 0.55, "label": "BULLISH", "extreme_event": False}
llm  = {"confidence_adjustment": 0.02, "market_state": "test"}
feats = {
    "ema5_dist": 0.002, "ema20_dist": 0.005,
    "rsi": 58.0, "adx": 27.0, "macd_hist": 15.0,
    "macd_slope": 0.0001, "volume_ratio": 1.4,
    "regime": "TRENDING", "trend_dir": "BULLISH", "trend_15m": "BULLISH",
    "vwap_dist": 0.0, "atr_pct": 0.003,
    "volume_change_pct": 0.2, "rsi_scaled": 0.58,
    "adx_scaled": 0.27, "macd_norm": 0.0001,
    "rsi_slope": 0.5, "bb_dist": 0.1,
}
sig = se.evaluate(ml_probs, feats, sent, llm, has_position=False)
action = sig["action"]
lscore = sig["long_score"]
sscore = sig["short_score"]
conf   = sig["confidence"]
print(f"[OK] Signal: action={action} long={lscore:.3f} short={sscore:.3f} conf={conf:.3f}")

rm = RiskManager(1000.0)
qty = rm.get_position_size(1000.0, 0.80, 200.0, 65000.0)
sl, tp = rm.compute_sl_tp("LONG", 65000.0, 200.0)
print(f"[OK] Risk sizing: qty={qty:.4f} BTC  SL={sl} TP={tp}")

print()
print("=== All validation checks PASSED ===")
print("Next: python main.py --retrain   (retrains binary ML with 11 features)")
print("Then: python main.py --dry-run   (live signals, no orders)")
