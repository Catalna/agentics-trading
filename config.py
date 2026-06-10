"""
config.py — Central configuration for the Scalping Expert Advisor.
All tuneable parameters live here. Load secrets from .env file.
"""

import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv(override=True)

# ─── Exchange ──────────────────────────────────────────────────────────────────
EXCHANGE_ID          = "binanceusdm"
TESTNET              = True
API_KEY              = os.getenv("BINANCE_API_KEY", "")
API_SECRET           = os.getenv("BINANCE_API_SECRET", "")
SYMBOL = "BTC/USDT:USDT"
LEVERAGE             = 100            # Extreme Scalping: kecil modal, besar notional
ORDER_TYPE           = "limit"        # Limit Order = Maker fee (0.02%), no slippage

# ─── Timeframes ────────────────────────────────────────────────────────────────
TF_MICRO             = "1m"           # For micro-trend and orderbook veto
TF_EXEC              = "5m"
TF_TREND             = "15m"
TF_SWING_1           = "1h"
TF_SWING_2           = "4h"
TF_MACRO             = "1d"
PREDICTION_HORIZON   = 4               # Candles ahead for label engineering

# ─── Data ──────────────────────────────────────────────────────────────────────
LOOKBACK_CANDLES_1M  = 50
LOOKBACK_CANDLES_5M  = 2000
LOOKBACK_CANDLES_15M = 1500
LOOKBACK_CANDLES_1H  = 500
LOOKBACK_CANDLES_4H  = 200
LOOKBACK_CANDLES_1D  = 100
TRAINING_DAYS        = 90              # Increased: more data → better model
CANDLE_LIMIT         = 1000

XGB_PARAMS = {
    "n_estimators":      600,          # More trees for richer features
    "max_depth":         6,            # Slightly deeper — more complex patterns
    "learning_rate":     0.015,        # Slower learn rate = better generalisation
    "subsample":         0.80,
    "colsample_bytree":  0.80,
    "gamma":             0.02,         # Lower gamma = more splits = more precision
    "min_child_weight":  2,            # Lower = catches rare setups
    "eval_metric":       "logloss",
    "random_state":      42,
    "n_jobs":            -1,
    "scale_pos_weight":  1,            # Re-balanced per class via sample_weight
    "tree_method":       "hist",       # GPU acceleration
    "device":            "cuda",       # Use RTX 5050
}
MODEL_PATH           = "models/xgb_model.json"
RETRAIN_INTERVAL_H   = 24

# Label classes
# ML predicts LONG (0) vs SHORT (1). HOLD emerges from insufficient confidence.
LABEL_LONG           = 0
LABEL_SHORT          = 1

# ─── ATR ───────────────────────────────────────────────────────────────────────
ATR_PERIOD           = 14

# ATR Targets (dipakai saat USE_PERCENTAGE_TGT = False)
TP_ATR_MULT          = 2.00           # TP: 2× ATR
SL_ATR_MULT          = 1.00           # SL: 1.5× ATR
TRAILING_ATR_MULT    = 0.50           # Trailing stop distance

# Fixed Percentage-Based Targets — AKTIF untuk scalping 5-menit
USE_PERCENTAGE_TGT   = True           # Pakai fixed % supaya AI cepat TP/SL di scalping
TP_PCT               = 0.0025         # 0.25% = profit cukup untuk cover maker fee 2 arah (0.04% total)
SL_PCT               = 0.0015         # 0.15% = SL dinaikkan sedikit agar tidak mudah tersentuh

# ─── Indicators ────────────────────────────────────────────────────────────────
EMA_FAST             = 5
EMA_SCALP            = 13
EMA_SLOW             = 20
RSI_PERIOD           = 14
LINREG_PERIOD        = 20
MACD_FAST            = 12
MACD_SLOW            = 26
MACD_SIGNAL          = 9
ADX_PERIOD           = 14
ADX_TREND_THRESHOLD  = 18             # Even lower — catch trends earlier
BB_PERIOD            = 20             # Bollinger Bands
BB_STD               = 2.0
VOLUME_RATIO_MIN     = 1.1            # Lowered for faster entry in scalp mode

# ─── Signal Architecture ───────────────────────────────────────────────────────
# Always produce LONG_SCORE vs SHORT_SCORE; pick the higher one.
# HOLD is NOT a signal — it's a position management state.
# Weights sum to 1.0.  Swarm replaces 10% from ML when enabled.
WEIGHT_ML            = 0.55           # ML engine — primary signal
WEIGHT_TECH          = 0.15           # Technical indicators — boosted for precision
WEIGHT_SENTIMENT     = 0.08           # Sentiment engine
WEIGHT_LLM           = 0.12           # Multi-LLM consensus — boosted
WEIGHT_SWARM         = 0.10           # Swarm agent consensus (14 agents, 4 layers)

# ─── Swarm Agent System ────────────────────────────────────────────────────────
SWARM_ENABLED        = True            # Enable swarm agent layer
SWARM_CONSENSUS_THRESHOLD = 0.55      # Min consensus confidence to count as signal

# ─── Confidence Thresholds ─────────────────────────────────────────────────────
CONF_NO_POSITION     = 0.52           # Agresif: entry mudah untuk scalping cepat
CONF_EXISTING        = 0.52           # Sama dengan entry — reversal instan
CONF_REVERSAL        = 0.52           # Bypass age filter jika sinyal kuat

# ─── Technical Entry Conditions ────────────────────────────────────────────────
# LONG bias: EMA5>EMA20, RSI 50–70, MACD bullish, ADX>20, volume>1.2×
# SHORT bias: EMA5<EMA20, RSI 30–50, MACD bearish, ADX>20, volume>1.2×
RSI_LONG_MIN         = 50
RSI_LONG_MAX         = 70
RSI_SHORT_MIN        = 30
RSI_SHORT_MAX        = 50

# ─── Risk Management ───────────────────────────────────────────────────────────
RISK_TIER_LOW        = 0.005          # 0.5%
RISK_TIER_MID        = 0.010          # 1.0%
RISK_TIER_HIGH       = 0.015          # 1.5%
MAX_DAILY_LOSS_PCT   = 0.02
MAX_DRAWDOWN_PCT     = 0.10
MIN_POSITION_NOTIONAL = 10.0
MAX_EFFECTIVE_LEVERAGE = 3.0          # Max single-trade notional size as a multiple of account balance
MARGIN_ALLOCATION_PCT = 0.05          # Margin allocated per trade (5% of balance used as margin)

# ─── Fee Context (otomatis disesuaikan engine) ────────────────────────────────
# Gunakan LIVE_FEE_PCT saat trading nyata (Limit Order = Maker fee)
# Backtest simulator sudah punya BT_MAKER_FEE sendiri
LIVE_MAKER_FEE_PCT   = 0.0002         # 0.02% Binance Maker (Limit Order)
LIVE_TAKER_FEE_PCT   = 0.0004         # 0.04% Binance Taker (Market Order - fallback)

# ─── Kelly Criterion (Rolling Window Position Sizing) ───────────────────────────
KELLY_WINDOW         = 7             # Jumlah trade terakhir untuk kalkulasi Kelly (rolling window)
KELLY_MAX_FRACTION   = 0.50          # Half-Kelly cap — max 50% dari full Kelly untuk safety
KELLY_MIN_MARGIN_PCT = 0.01          # Min margin 1% saat kondisi sangat buruk
KELLY_MAX_MARGIN_PCT = 0.10          # Max margin 10% saat kondisi sangat bagus
KELLY_MIN_TRADES     = 3             # Min trade di window sebelum Kelly aktif (pakai default jika belum)

# ─── Anti-Whipsaw Reversal Filter ──────────────────────────────────────────────
REVERSAL_MIN_AGE_MIN = 5             # Dinaikkan jadi 5 menit untuk mencegah whipsaw bolak-balik (hemat fee)
REVERSAL_BYPASS_CONF = 0.85          # Dinaikkan jadi 0.85 agar bypass hanya terjadi saat sinyal sangat kuat

# ─── LLM ───────────────────────────────────────────────────────────────────────
OLLAMA_URL           = "http://localhost:11434"
OLLAMA_MODEL         = "llama3.1:latest"     # Primary / fallback model
LLM_TIMEOUT_S        = 120     # Timeout per LLM call (seconds) — increased for local GPU
LLM_MAX_INFLUENCE    = 0.06                  # Slight boost to LLM influence

# Multi-LLM Voting — 3 installed Ollama models with contribution weights
# Weights must sum to 1.0.  Larger/smarter models get higher weight.
OLLAMA_MODELS = {
    "deepseek-r1:7b":  0.50,   # Chief Supervisor — final decision
    "qwen2.5:7b":      0.30,   # Technical Manager
    "mistral:7b":      0.20,   # Sentiment Manager
}
MULTI_LLM_ENABLED    = True    # Set False to revert to single-model mode
BACKTEST_SKIP_LLM    = True    # Skip LLM calls in backtest for speed (use neutral)

# ─── Sentiment ─────────────────────────────────────────────────────────────────
# 7 pure-RSS/API sources — no credentials required
SENT_W_FNG           = 0.20   # Fear & Greed (most reliable)
SENT_W_COINTELEGRAPH = 0.20   # CoinTelegraph RSS
SENT_W_DECRYPT       = 0.15   # Decrypt RSS
SENT_W_BINANCE_ANN   = 0.15   # Binance Announcements RSS
SENT_W_BITCOINCOM    = 0.15   # Bitcoin.com RSS
SENT_W_CRYPTOPOTATO  = 0.10   # CryptoPotato RSS
SENT_W_YOUTUBE       = 0.05   # YouTube channel RSS

# Refresh intervals
SENT_REFRESH_FNG       = 3600
SENT_REFRESH_RSS       = 300
SENT_REFRESH_BINANCE   = 300
SENT_REFRESH_YOUTUBE   = 600

# API URLs
FNG_URL              = "https://api.alternative.me/fng/?limit=1"
COINTELEGRAPH_RSS    = "https://cointelegraph.com/rss"
DECRYPT_RSS          = "https://decrypt.co/feed"
BINANCE_ANN_RSS      = "https://www.binance.com/en/support/announcement/rss"
BITCOINCOM_RSS       = "https://news.bitcoin.com/feed/"
CRYPTOPOTATO_RSS     = "https://cryptopotato.com/feed/"

YOUTUBE_RSS_FEEDS    = [
    "https://www.youtube.com/feeds/videos.xml?channel_id=UCqK_GSMbpiV8spgD3ZGloSw",  # Coin Bureau
    "https://www.youtube.com/feeds/videos.xml?channel_id=UCbLhGKVY-bJPcawebgtNfbw",  # Altcoin Daily
    "https://www.youtube.com/feeds/videos.xml?channel_id=UCN9Nj4tjXbVTLYWN0NKpLqg",  # Crypto Banter
]
YOUTUBE_MAX_ITEMS    = 5

BULLISH_KEYWORDS = [
    "bullish", "breakout", "surge", "etf approval", "institutional buying",
    "accumulation", "record inflow", "all-time high", "ath", "rally",
    "adoption", "positive", "recovery", "moon", "pump", "soar",
]
BEARISH_KEYWORDS = [
    "hack", "liquidation", "lawsuit", "recession", "ban", "dump",
    "selloff", "outflow", "crash", "collapse", "regulation", "fud",
    "negative", "bear", "drop", "plunge", "sell-off", "fraud",
]
EXTREME_EVENT_KEYWORDS = [
    "exchange hack", "exchange hacked", "etf rejected", "etf denial",
    "mass liquidation", "flash crash", "federal reserve", "fed meeting",
    "cpi release", "interest rate", "emergency", "ban crypto",
    "sec lawsuit", "market halt", "trading halted", "black swan",
]

SENT_LABEL_VERY_BULLISH  = 0.70
SENT_LABEL_BULLISH       = 0.55
SENT_LABEL_NEUTRAL_HIGH  = 0.50
SENT_LABEL_NEUTRAL_LOW   = 0.45
SENT_LABEL_BEARISH       = 0.35

SENT_ADJ_VERY_BULLISH  = +0.05
SENT_ADJ_BULLISH       = +0.02
SENT_ADJ_NEUTRAL       =  0.00
SENT_ADJ_BEARISH       = -0.02
SENT_ADJ_VERY_BEARISH  = -0.05
SENT_ADJ_EXTREME_EVENT = -0.15

SENTIMENT_CACHE_PATH  = "data/sentiment_cache.json"

# ─── Backtest ──────────────────────────────────────────────────────────────────
BT_TAKER_FEE         = 0.0004
BT_MAKER_FEE         = 0.0002
BT_SLIPPAGE          = 0.0
BT_FUNDING_FEE       = 0.0001
BT_FUNDING_INTERVAL  = 8 * 60 * 60
BT_TRAIN_DAYS        = 14
BT_TEST_DAYS         = 7
BT_TOTAL_DAYS        = 30

# ─── Paths ─────────────────────────────────────────────────────────────────────
LOG_DIR              = "logs"
DATA_DIR             = "data"
MODEL_DIR            = "models"
DB_PATH              = "logs/trading.db"
LOG_FILE             = "logs/ea.log"

# ─── Loop Timing ───────────────────────────────────────────────────────────────
MAIN_LOOP_SLEEP_S    = 2              # Faster loop — 2s instead of 3s
COOLDOWN_AFTER_TRADE = 0             # Zero cooldown — immediate re-entry for HFT mode

# ─── Exchange & Execution (Binance) ────────────────────────────────────────────
BINANCE_API_KEY      = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY   = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET          = True          # Set True for paper trading on Binance Testnet
DEFAULT_LEVERAGE     = 100           # MAX 100x — Binance USDT-M Futures

# ─── Risk Management & SL/TP for 100x ─────────────────────────────────────────
# At 100x leverage, liquidation occurs at ~1% from entry.
# SL must be < liquidation to avoid getting wiped.
# Rule: SL = 0.3% | TP = 0.6% | RR = 2:1
RISK_PER_TRADE_PCT   = 0.01          # Risk 1% of account balance per trade
DEFAULT_SL_PCT       = 0.003         # 0.3% SL — safe buffer before 100x liquidation
DEFAULT_TP_PCT       = 0.006         # 0.6% TP — 2:1 RR ratio
MAX_OPEN_POSITIONS   = 1
MIN_CONFIDENCE       = 0.65          # SNIPER + EA MODE: 70% confidence for HTF-aligned momentum

CLAUDE_API_KEY       = ""
USE_CLAUDE_VALIDATION = False        # If True, Claude validates final DeepSeek decision