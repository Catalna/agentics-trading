import time
import json
import logging
import pandas as pd

import config
from core.logger import setup_logger
import xgboost as xgb
from core.feature_engine import FeatureEngine
from core.agents.llm_orchestrator import LLMOrchestrator
from core.executor import TradeExecutor

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

logger = setup_logger("MainLoop")

# ─── Feature column list (source of truth from training) ──────────────────────
_FEATURE_COLS_PATH = "models/feature_columns.json"


def load_feature_cols(model=None) -> list:
    """
    Loads the feature column list saved by train_models.py.
    Fallback: extracts feature names directly from the XGBoost booster.
    """
    try:
        with open(_FEATURE_COLS_PATH, "r") as f:
            cols = json.load(f)
            logger.info(f"Loaded {len(cols)} feature columns from {_FEATURE_COLS_PATH}")
            return cols
    except FileNotFoundError:
        # Fallback: extract from model booster (XGBoost stores feature names internally)
        if model is not None:
            try:
                cols = model.get_booster().feature_names
                if cols:
                    logger.info(
                        f"feature_columns.json not found — using {len(cols)} feature names "
                        f"embedded in the model booster."
                    )
                    return list(cols)
            except Exception as e:
                logger.warning(f"Could not extract feature names from booster: {e}")

        logger.warning(
            f"Feature columns file not found at '{_FEATURE_COLS_PATH}' and model has no "
            "embedded feature names. ML predictions will be skipped. "
            "Run `python train_models.py` to generate the file."
        )
        return []


# ─── DataEngine ────────────────────────────────────────────────────────────────
class DataEngine:
    """Fetches real-time market data across multiple timeframes."""

    def __init__(self, symbol: str = config.SYMBOL):
        self.symbol = symbol
        if not CCXT_AVAILABLE:
            self.exchange = None
            return

        # Use binanceusdm (USDT-M Futures), consistent with config.EXCHANGE_ID
        exchange_class = getattr(ccxt, config.EXCHANGE_ID, ccxt.binanceusdm)
        # Do not pass API keys to DataEngine so it can fetch public data (OHLCV) freely
        self.exchange = exchange_class({
            "enableRateLimit": True,
        })

        # Activate testnet / sandbox mode if configured
        if config.TESTNET:
            # Bypass CCXT's deprecation exception for binanceusdm testnet
            self.exchange.urls['api']['fapiPublic'] = 'https://testnet.binancefuture.com/fapi/v1'
            self.exchange.urls['api']['fapiPrivate'] = 'https://testnet.binancefuture.com/fapi/v1'
            self.exchange.urls['api']['fapiPublicV2'] = 'https://testnet.binancefuture.com/fapi/v2'
            self.exchange.urls['api']['fapiPrivateV2'] = 'https://testnet.binancefuture.com/fapi/v2'
            logger.info("DataEngine: Binance USDT-M Futures TESTNET mode aktif.")

    def _fetch_ohlcv(self, timeframe: str, limit: int) -> pd.DataFrame:
        """Internal: fetch OHLCV for one timeframe and return as DataFrame."""
        if not self.exchange:
            logger.warning("Exchange not available.")
            return pd.DataFrame()
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            logger.error(f"Error fetching {timeframe} data: {e}")
            return pd.DataFrame()

    def fetch_latest_data(self, timeframe: str = "5m", limit: int = 100) -> pd.DataFrame:
        """Single-TF convenience wrapper (used by LLM market summary)."""
        return self._fetch_ohlcv(timeframe, limit)

    def fetch_multi_tf(self) -> dict:
        """
        Fetches data for all three timeframes required by the ML model.
        Returns a dict with keys '5m', '1h', '1d'.

        Candle counts chosen to ensure enough history for all rolling indicators:
          5m  → 500 candles (~41 h) — EMA34 + LR20 + BB20 + volume_ma20
          1h  → 300 candles (~12 d) — EMA144 warm-up
          1d  → 150 candles (~5 mo) — EMA377 warm-up
        """
        logger.info("Fetching multi-TF market data (5m / 1h / 1d)...")
        df_5m = self._fetch_ohlcv("5m",  limit=500)
        df_1h = self._fetch_ohlcv("1h",  limit=300)
        df_1d = self._fetch_ohlcv("1d",  limit=150)

        if df_5m.empty or df_1h.empty or df_1d.empty:
            logger.warning("One or more timeframes returned empty data.")
        else:
            logger.info(
                f"Data fetched — 5m:{len(df_5m)} | 1h:{len(df_1h)} | 1d:{len(df_1d)} rows"
            )

        return {"5m": df_5m, "1h": df_1h, "1d": df_1d}


# ─── Model Loader ──────────────────────────────────────────────────────────────
def load_xgb_model(path: str):
    """
    Loads an XGBClassifier saved via XGBClassifier.save_model().
    Using XGBClassifier (not xgb.Booster) ensures predict() returns
    class indices directly (0=HOLD, 1=LONG, 2=SHORT).
    """
    try:
        model = xgb.XGBClassifier()
        model.load_model(path)
        logger.info(f"XGBoost model loaded from '{path}'")
        return model
    except Exception as e:
        logger.warning(f"Failed to load XGBoost model from '{path}': {e}")
        return None

# ─── Main Loop ────────────────────────────────────────────────────────────────
def run_trading_bot():
    logger.info("Initializing AI Crypto Futures Trading Bot v4.2")

    # 1. Initialize Components
    data_engine    = DataEngine()
    feature_engine = FeatureEngine()

    logger.info("Initializing XGBoost ML model...")
    xgb_model = load_xgb_model("models/xgb_aggressive_v4.json")

    # Load the exact feature list used during training.
    # Fallback: read from model's booster (XGBoost stores them internally).
    feature_cols = load_feature_cols(model=xgb_model)

    llm_orchestrator = LLMOrchestrator()
    executor         = TradeExecutor()

    # Capture starting balance for PnL / ROI calculation
    initial_balance = executor.get_live_balance()
    if initial_balance > 0:
        logger.info(f"Starting Balance : ${initial_balance:,.2f} USDT")

    logger.info("All components initialized. Starting Main Loop...")
    start_time = time.time()

    while True:
        try:
            print("\n" + "="*60)
            logger.info("NEW TRADING CYCLE STARTED")
            print("="*60)

            # 2. Fetch Multi-TF Data
            tf_data = data_engine.fetch_multi_tf()
            df_5m   = tf_data["5m"]
            df_1h   = tf_data["1h"]
            df_1d   = tf_data["1d"]

            if df_5m.empty:
                logger.warning("No 5m data received. Retrying in 10s...")
                time.sleep(10)
                continue

            current_price = df_5m["close"].iloc[-1]
            print(f"[DATA]  Fetched 5m: {len(df_5m)} | 1h: {len(df_1h)} | 1d: {len(df_1d)} rows")
            print(f"[PRICE] {config.SYMBOL} = ${current_price:,.2f}")
            print("-" * 60)

            # 3. Multi-TF Feature Engineering
            #    merge_multi_tf aligns 1h/1d features onto the 5m index (ffill)
            #    and appends confluence_score + tf_align_score.
            df_features = feature_engine.merge_multi_tf(df_5m, df_1h, df_1d)

            # Single row for inference
            latest_row = df_features.iloc[[-1]]
            latest_feats = df_features.iloc[-1].to_dict()

            # 4. ML Predictions
            ml_preds = {"xgboost": "NEUTRAL"}

            if xgb_model and feature_cols:
                try:
                    # Validate all training features are present
                    missing = [c for c in feature_cols if c not in df_features.columns]
                    if missing:
                        logger.warning(
                            f"{len(missing)} feature(s) missing in live data: "
                            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
                        )
                    else:
                        # Pass only the exact training feature columns — no OHLCV raw cols
                        X_live = latest_row[feature_cols].fillna(0)
                        pred_class = int(xgb_model.predict(X_live)[0])
                        if pred_class == 1:
                            ml_preds["xgboost"] = "LONG"
                            icon = "[UP]"
                        elif pred_class == 2:
                            ml_preds["xgboost"] = "SHORT"
                            icon = "[DOWN]"
                        else:
                            ml_preds["xgboost"] = "NEUTRAL"
                            icon = "[WAIT]"
                            
                        print(f"[XGBOOST] PREDICTION: {icon} {ml_preds['xgboost']} (Class {pred_class})")
                except Exception as e:
                    logger.warning(f"ML Prediction Error: {e}")

            elif not feature_cols:
                logger.warning(
                    "Skipping ML prediction — run `python train_models.py` to generate "
                    "models/feature_columns.json first."
                )
            print("-" * 60)



            # 5. LLM Orchestration
            market_summary = (
                f"Symbol: {config.SYMBOL}\n"
                f"Current Price: {current_price:.2f}\n"
                f"RSI (5m): {latest_feats.get('rsi_5m', 'N/A'):.2f}\n"
                f"MACD Hist (5m): {latest_feats.get('macd_hist_5m', 'N/A'):.4f}\n"
                f"EMA Align (5m): {latest_feats.get('ema_align_score_5m', 'N/A')}\n"
                f"RSI (1h): {latest_feats.get('rsi_1h', 'N/A'):.2f}\n"
                f"EMA Align (1h): {latest_feats.get('ema_align_score_1h', 'N/A')}\n"
                f"Confluence Score: {latest_feats.get('confluence_score', 'N/A'):.3f}\n"
                f"TF Align Score: {latest_feats.get('tf_align_score', 'N/A'):.3f}\n"
                f"Volume Ratio (5m): {latest_feats.get('volume_ratio_5m', 'N/A'):.2f}\n"
            )

            llm_result = llm_orchestrator.orchestrate(market_summary, ml_preds)

            # 6. Final Decision & Execution
            final_decision = llm_result.get("chief", {})

            if config.USE_CLAUDE_VALIDATION and config.CLAUDE_API_KEY:
                logger.info("Claude validation enabled (not yet implemented). Skipping.")

            executor.execute_trade(final_decision, current_price)

            # 7. Sleep for next cycle
            elapsed = time.time() - start_time
            logger.info(
                f"Cycle complete. Elapsed: {elapsed:.0f}s. "
                f"Sleeping {config.MAIN_LOOP_SLEEP_S}s..."
            )

            if elapsed >= 1800:
                logger.info("30 Minutes Live Run Complete!")
                break

            time.sleep(config.MAIN_LOOP_SLEEP_S)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(10)

    # ── Final Report ──────────────────────────────────────────────────────────
    final_balance = executor.get_live_balance()

    if initial_balance > 0 and final_balance > 0:
        pnl = final_balance - initial_balance
        roi = (pnl / initial_balance) * 100
        pnl_str = f"${pnl:+,.2f}"
        roi_str = f"{roi:+.2f}%"
        bal_str = f"${final_balance:,.2f}"
    else:
        pnl_str = "N/A"
        roi_str = "N/A"
        bal_str = f"${final_balance:,.2f}" if final_balance > 0 else "N/A"

    elapsed_min = (time.time() - start_time) / 60

    report = (
        "\n=====================================\n"
        "         LIVE TRADING REPORT         \n"
        "=====================================\n"
        f"Mode                 : {'TESTNET' if config.USE_TESTNET else 'MAINNET'}\n"
        f"Run Duration         : {elapsed_min:.1f} min\n"
        "-------------------------------------\n"
        f"Starting Balance     : ${initial_balance:,.2f}\n"
        f"Final Balance        : {bal_str}\n"
        f"Net PnL              : {pnl_str}\n"
        f"ROI                  : {roi_str}\n"
        "-------------------------------------\n"
        f"Total Trades Sent    : {len(executor.trade_history)}\n"
        "====================================="
    )
    logger.info(report)


if __name__ == "__main__":
    run_trading_bot()
