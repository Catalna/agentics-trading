"""
generate_synthetic_data.py — High-Fidelity Noisy Synthetic Market Generator.

Generates 100,000 candles representing 5m price action across 4 regimes:
  1. Ranging/Mean-Reverting (Ornstein-Uhlenbeck process)
  2. Strong Bullish Trend (positive drift, high volume on buy candles)
  3. Strong Bearish Trend (negative drift, high volume on sell candles)
  4. Breakout Spikes (explosive momentum in random directions)

Injects organic noise into the price delta and High/Low spreads to force the 
XGBoost ML model to generalize better and strictly prevent overfitting.
"""

import os
import time
import numpy as np
import pandas as pd

import config
from core.feature_engine import FeatureEngine
from core.ml_engine import _build_labels

def main():
    print("=" * 60)
    print("      NOISY HIGH-FIDELITY SYNTHETIC PRICE GENERATOR")
    print("=" * 60)
    print(f"Generating 100,000 candles of 5m trading data with organic noise injection...")

    # Set random seed for reproducibility
    np.random.seed(42)

    n_samples = 100000
    p_init = 65000.0
    
    # ─── Markov Regime Transition Matrix ──────────────────────────────────────
    # States: 0=Ranging, 1=Bull Trend, 2=Bear Trend, 3=Breakout
    transition_matrix = np.array([
        [0.85, 0.06, 0.06, 0.03],  # From Ranging
        [0.08, 0.90, 0.01, 0.01],  # From Bull Trend
        [0.08, 0.01, 0.90, 0.01],  # From Bear Trend
        [0.40, 0.30, 0.30, 0.00]   # From Breakout (immediately transitions out)
    ])

    current_state = 0
    price = p_init
    
    # Track states and parameters
    prices = [price]
    states = [current_state]
    volumes = []
    highs = []
    lows = []
    opens = []
    closes = []
    
    # For mean reversion baseline in ranging state
    baseline = price
    
    start_time = time.time()
    
    for t in range(n_samples):
        # 1. State transition
        current_state = np.random.choice([0, 1, 2, 3], p=transition_matrix[current_state])
        states.append(current_state)
        
        # 2. Determine price drift & volatility based on regime
        # base volatility as percentage of price
        vol = 0.0006  
        
        if current_state == 0:  # RANGING
            # Mean reversion to local baseline
            drift = -0.06 * (price - baseline) / baseline
            volatility = vol * 0.8
            base_vol = np.random.exponential(120) + 20
            vol_multiplier = 0.9
        elif current_state == 1:  # BULL TREND
            drift = 0.0012  # upward drift
            volatility = vol * 1.1
            base_vol = np.random.exponential(250) + 50
            vol_multiplier = 1.3
        elif current_state == 2:  # BEAR TREND
            drift = -0.0014  # downward drift
            volatility = vol * 1.3
            base_vol = np.random.exponential(300) + 60
            vol_multiplier = 1.4
        else:  # BREAKOUT SPIKE (3)
            direction = np.random.choice([1, -1], p=[0.5, 0.5])
            drift = direction * 0.008  # massive price jump
            volatility = vol * 2.5
            base_vol = np.random.exponential(800) + 200
            vol_multiplier = 3.5
            
        # 3. Simulate price change using Brownian Motion + EXTRA ORGANIC NOISE
        # Injects random white noise (representing market noise/random fills)
        market_noise = np.random.normal(0, vol * 0.6)  # 60% of baseline volatility is purely random noise
        price_change_pct = drift + volatility * np.random.normal(0, 1) + market_noise
        
        # Clip crazy outliers
        price_change_pct = np.clip(price_change_pct, -0.035, 0.035)
        
        open_price = price
        close_price = price * (1 + price_change_pct)
        price = close_price
        
        # 4. Generate High & Low with realistic + noisy ATR spreads
        atr_level = close_price * (0.0012 + np.random.exponential(0.0004))
        # Random spread multiplier to simulate fluctuating market bid-ask spreads
        spread_noise_mult = np.random.uniform(0.75, 1.45)
        
        high_price = max(open_price, close_price) + abs(np.random.normal(0, atr_level * 0.45)) * spread_noise_mult
        low_price = min(open_price, close_price) - abs(np.random.normal(0, atr_level * 0.45)) * spread_noise_mult
        
        # 5. Volume construction
        volume = base_vol * vol_multiplier
        if current_state in (1, 3) and close_price > open_price:
            volume *= 1.4
        elif current_state in (2, 3) and close_price < open_price:
            volume *= 1.5
            
        opens.append(open_price)
        closes.append(close_price)
        highs.append(high_price)
        lows.append(low_price)
        volumes.append(volume)
        prices.append(price)
        
        # Adjust local ranging baseline slowly
        if current_state != 0 and np.random.random() < 0.05:
            baseline = price

    # Create raw DataFrame
    start_ts = int(time.time()) - (n_samples * 5 * 60)
    timestamps = [start_ts + (i * 5 * 60) for i in range(n_samples)]
    
    df_raw = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })

    print(f"Candles generated in {time.time() - start_time:.2f} seconds.")
    print(f"Price range: {df_raw['close'].min():.2f} - {df_raw['close'].max():.2f}")
    
    # Save raw OHLCV to CSV
    os.makedirs("data", exist_ok=True)
    csv_path = "data/synthetic_ohlcv.csv"
    df_raw.to_csv(csv_path, index=False)
    print(f"Saved raw price candles -> {csv_path}")
    
    print("\nAnalyzing dataset quality with features & labels...")
    fe = FeatureEngine()
    df_feat = fe.compute(df_raw)
    df_feat = df_feat.dropna().copy()
    
    labels = _build_labels(df_feat)
    df_feat["label"] = labels
    
    # Drop neutral/NaN rows
    df_clean = df_feat.dropna(subset=["label"]).copy()
    n_total = len(df_clean)
    
    n_long = int((df_clean["label"] == config.LABEL_LONG).sum())
    n_short = int((df_clean["label"] == config.LABEL_SHORT).sum())
    
    print("-" * 60)
    print(f"Total labeled samples   : {n_total}")
    print(f"LONG Setup samples (0)  : {n_long} ({n_long/n_total*100:.1f}%)")
    print(f"SHORT Setup samples (1) : {n_short} ({n_short/n_total*100:.1f}%)")
    print("-" * 60)
    print("Generator successfully completed and ready for ML training!")

if __name__ == "__main__":
    main()
