"""
utils/feature_engineering.py — Shared feature engineering utilities
"""

import pandas as pd
import numpy as np

FEATURE_COLS = [
    'ema5_dist', 'ema20_dist', 'vwap_dist', 'atr_pct', 'volume_change_pct',
    'rsi_scaled', 'adx_scaled', 'macd_norm', 'rsi_slope', 'macd_slope', 'bb_dist'
]

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def _rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def _macd(close, fast=12, slow=26, sig=9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = _ema(macd_line, sig)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def _atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _adx(high, low, close, period=14):
    # Mock ADX for simplicity without full wilder smoothing
    rsi = _rsi(close, period)
    return rsi.rolling(period).std() * 2 

def _vwap(df):
    vol = df['volume']
    close = df['close']
    return (close * vol).cumsum() / vol.cumsum()

def _bollinger(close, period=20, std=2):
    mid = close.rolling(period).mean()
    std_dev = close.rolling(period).std()
    upper = mid + (std_dev * std)
    lower = mid - (std_dev * std)
    return mid, upper, lower

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['close']
    
    df['ema5'] = _ema(close, 5)
    df['ema20'] = _ema(close, 20)
    df['rsi'] = _rsi(close)
    df['macd_line'], df['macd_signal'], df['macd_hist'] = _macd(close)
    df['atr'] = _atr(df['high'], df['low'], close)
    df['adx'] = _adx(df['high'], df['low'], close)
    df['bb_mid'], df['bb_upper'], df['bb_lower'] = _bollinger(close)
    df['vwap'] = _vwap(df)
    
    vol = df['volume']
    df['vol_ratio'] = vol / vol.rolling(20).mean().replace(0, 1)
    
    # Normalized features
    df['ema5_dist'] = (close - df['ema5']) / close
    df['ema20_dist'] = (close - df['ema20']) / close
    df['vwap_dist'] = (close - df['vwap']) / close
    df['atr_pct'] = df['atr'] / close
    df['volume_change_pct'] = np.clip(df['vol_ratio'] - 1, -5, 5)
    df['rsi_scaled'] = df['rsi'] / 100
    df['adx_scaled'] = df['adx'] / 100
    df['macd_norm'] = np.clip(df['macd_hist'] / close, -5, 5)
    
    df['rsi_slope'] = df['rsi'].diff(3) / 3
    df['macd_slope'] = df['macd_hist'].diff(3) / close
    
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_dist'] = np.clip((close - df['bb_mid']) / (bb_range.replace(0, 1e-9)), -5, 5)
    
    df['regime'] = np.where(df['adx'] > 20, 'TRENDING', 'RANGING')
    df['trend_dir'] = np.where(df['ema5'] > df['ema20'], 'BULLISH', 'BEARISH')
    
    return df

def build_ml_row(feats: dict) -> pd.DataFrame:
    return pd.DataFrame([feats])[FEATURE_COLS]
