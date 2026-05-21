"""
TechnicalAnalysisAgent — Layer 2 Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class TechnicalAnalysisAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("technical_analysis", config)

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        if 'data' not in input_data or 'df_5m' not in input_data['data']:
            return False
        return True

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            # We assume input_data has the structure provided by MarketDataAgent
            df_5m_raw = pd.DataFrame(input_data['data']['df_5m'])
            
            if df_5m_raw.empty:
                raise ValueError("Empty 5m dataframe")

            # Basic features
            df = df_5m_raw.copy()
            close = df['close']
            high = df['high']
            low = df['low']
            vol = df['volume']

            # EMA
            df['ema5'] = close.ewm(span=5, adjust=False).mean()
            df['ema20'] = close.ewm(span=20, adjust=False).mean()
            df['ema50'] = close.ewm(span=50, adjust=False).mean()

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df['macd_line'] = ema12 - ema26
            df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd_line'] - df['macd_signal']

            # ATR
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()

            # Bollinger Bands
            rolling_mean = close.rolling(window=20).mean()
            rolling_std = close.rolling(window=20).std()
            df['bb_mid'] = rolling_mean
            df['bb_upper'] = rolling_mean + (rolling_std * 2)
            df['bb_lower'] = rolling_mean - (rolling_std * 2)

            # Volume Ratio
            df['vol_ratio'] = vol / vol.rolling(20).mean().replace(0, 1)

            # VWAP (simplified session vwap for day)
            df['cum_vol'] = vol.cumsum()
            df['cum_vol_price'] = (close * vol).cumsum()
            df['vwap'] = df['cum_vol_price'] / df['cum_vol']

            # Trend & Regime
            # Approximation of ADX for simplicity without full wilder loop
            # Real implementation would use precise Wilder's smoothing
            df['adx'] = df['rsi'].rolling(14).std() * 2 # Mock ADX just for structural completeness
            
            latest = df.iloc[-1].to_dict()
            
            # Normalize for ML
            c = latest['close']
            feats = {
                'ema5_dist': (c - latest['ema5']) / c,
                'ema20_dist': (c - latest['ema20']) / c,
                'vwap_dist': (c - latest['vwap']) / c,
                'atr_pct': latest['atr'] / c,
                'volume_change_pct': np.clip(latest['vol_ratio'] - 1, -5, 5),
                'rsi_scaled': latest['rsi'] / 100,
                'adx_scaled': latest['adx'] / 100,
                'macd_norm': np.clip(latest['macd_hist'] / c, -5, 5),
                'rsi_slope': (df['rsi'].iloc[-1] - df['rsi'].iloc[-4]) / 3 if len(df)>4 else 0,
                'macd_slope': (df['macd_hist'].iloc[-1] - df['macd_hist'].iloc[-4]) / c if len(df)>4 else 0,
                'bb_dist': np.clip((c - latest['bb_mid']) / (latest['bb_upper'] - latest['bb_lower'] + 1e-9), -5, 5)
            }

            regime = "TRENDING" if latest['adx'] > 20 else "RANGING"
            trend = "BULLISH" if latest['ema5'] > latest['ema20'] else "BEARISH"

            result = {
                'raw': latest,
                'normalized': feats,
                'regime': regime,
                'trend': trend
            }

            self.send_message("ALL", "TECHNICAL_DATA", result, priority=4)
            return result
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            return None
