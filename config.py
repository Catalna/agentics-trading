"""
config.py — Central configuration for the Scalping Expert Advisor.
All tuneable parameters live here. Load secrets from .env file.
"""

import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

# ─── Exchange ──────────────────────────────────────────────────────────────────
EXCHANGE_ID          = "binanceusdm"
TESTNET              = True
API_KEY              = os.getenv("BINANCE_API_KEY", "")
API_SECRET           = os.getenv("BINANCE_API_SECRET", "")
SYMBOL = "BTC/USDT:USDT"
LEVERAGE             = 50
ORDER_TYPE           = "market"         # "market" for instant fills (open position), "limit" for no-slippage orderbook entries

# ─── Timeframes ────────────────────────────────────────────────────────────────
TF_EXEC              = "5m"
TF_TREND             = "15m"
PREDICTION_HORIZON   = 4               # Candles ahead for label engineering

# ─── Data ──────────────────────────────────────────────────────────────────────
LOOKBACK_CANDLES_5M  = 2000
LOOKBACK_CANDLES_15M = 1500
TRAINING_DAYS        = 90              # Increased: more data → better model
CANDLE_LIMIT         = 1000

# ─── ML / XGBoost ──────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":      400,          # More trees for richer features
    "max_depth":         5,
    "learning_rate":     0.02,
    "subsample":         0.75,
    "colsample_bytree":  0.75,
    "gamma":             0.05,
    "min_child_weight":  3,
    "eval_metric":       "logloss",
    "random_state":      42,
    "n_jobs":            -1,
}
MODEL_PATH           = "models/xgb_model.json"
RETRAIN_INTERVAL_H   = 24

# Label classes
# ML predicts LONG (0) vs SHORT (1). HOLD emerges from insufficient confidence.
LABEL_LONG           = 0
LABEL_SHORT          = 1

# ─── ATR ───────────────────────────────────────────────────────────────────────
ATR_PERIOD           = 14

# Aggressive TP system (100% target)
TP_ATR_MULT          = 0.50           # TP: close 100% of position (HFT scalp)
SL_ATR_MULT          = 0.20           # Dynamic SL (HFT scalp)

# Fixed Percentage-Based Targets (Alternative to ATR for ultra-fast scalping)
USE_PERCENTAGE_TGT   = True           # Set to True for fixed percentage-based scalp targets
TP_PCT               = 0.0012         # 0.12% price change (approx 6.0% ROI at 50x leverage)
SL_PCT               = 0.0005         # 0.05% price change (approx 2.5% ROI at 50x leverage)

# ─── Indicators ────────────────────────────────────────────────────────────────
EMA_FAST             = 5
EMA_SLOW             = 20
RSI_PERIOD           = 14
MACD_FAST            = 12
MACD_SLOW            = 26
MACD_SIGNAL          = 9
ADX_PERIOD           = 14
ADX_TREND_THRESHOLD  = 20             # Lowered from 25 to catch more trends
BB_PERIOD            = 20             # Bollinger Bands
BB_STD               = 2.0
VOLUME_RATIO_MIN     = 1.2            # Min volume ratio for entry confirmation

# ─── Signal Architecture ───────────────────────────────────────────────────────
# Always produce LONG_SCORE vs SHORT_SCORE; pick the higher one.
# HOLD is NOT a signal — it's a position management state.
# Weights sum to 1.0.  Swarm replaces 10% from ML when enabled.
WEIGHT_ML            = 0.60           # ML engine (reduced by 10% to make room for swarm)
WEIGHT_TECH          = 0.10           # Technical indicators
WEIGHT_SENTIMENT     = 0.10           # Sentiment engine
WEIGHT_LLM           = 0.10           # Multi-LLM consensus
WEIGHT_SWARM         = 0.10           # Swarm agent consensus (14 agents, 4 layers)

# ─── Swarm Agent System ────────────────────────────────────────────────────────
SWARM_ENABLED        = True            # Enable swarm agent layer
SWARM_CONSENSUS_THRESHOLD = 0.55      # Min consensus confidence to count as signal

# ─── Confidence Thresholds ─────────────────────────────────────────────────────
CONF_NO_POSITION     = 0.52           # Min score to enter when flat (reduced to 0.52 for ultra-aggressive MT4-EA style)
CONF_EXISTING        = 0.52           # Min opposite score to reverse (set to same as entry for instant reversal)
CONF_REVERSAL        = 0.52           # Min score to immediately reverse (set to same as entry for instant reversal)

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
MAX_EFFECTIVE_LEVERAGE = 2.0          # Max single-trade notional size as a multiple of account balance (safe for new/demo accounts)
MARGIN_ALLOCATION_PCT = 0.05          # Margin allocated per trade (e.g. 0.05 = 5% of balance used as margin)

# ─── LLM ───────────────────────────────────────────────────────────────────────
OLLAMA_URL           = "http://localhost:11434"
OLLAMA_MODEL         = "qwen2.5:7b"          # Primary / fallback model
LLM_TIMEOUT_S        = 15
LLM_MAX_INFLUENCE    = 0.05

# Multi-LLM Voting — all installed Ollama models with contribution weights
# Weights must sum to 1.0.  Larger/smarter models get higher weight.
OLLAMA_MODELS = {
    "qwen2.5:7b":      0.40,   # Primary analyst — highest accuracy
    "llama3.1:latest": 0.35,   # Secondary analyst — strong reasoning
    "qwen2.5:3b":      0.15,   # Fast lightweight voter
    "llama3.2:3b":     0.10,   # Tie-breaker / speed voter
}
MULTI_LLM_ENABLED   = True    # Set False to revert to single-model mode

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
BT_SLIPPAGE          = 0.0005
BT_FUNDING_FEE       = 0.0001
BT_FUNDING_INTERVAL  = 8 * 60 * 60
BT_TRAIN_DAYS        = 30
BT_TEST_DAYS         = 7
BT_TOTAL_DAYS        = 90

# ─── Paths ─────────────────────────────────────────────────────────────────────
LOG_DIR              = "logs"
DATA_DIR             = "data"
MODEL_DIR            = "models"
DB_PATH              = "logs/trading.db"
LOG_FILE             = "logs/ea.log"

# ─── Loop Timing ───────────────────────────────────────────────────────────────
MAIN_LOOP_SLEEP_S    = 3
COOLDOWN_AFTER_TRADE = 0              # Zero cooldown — immediate re-entry for HFT mode
