# MODELS.md
# Spesifikasi ML Models — AI Crypto Futures Trading Bot v4.2
> Dokumen ini mendefinisikan arsitektur, feature engineering, training pipeline,
> dan inference pipeline untuk seluruh ML Models dalam sistem.
> Dibaca bersama: `AGENTS.md`, `INDICATORS.md`

---

## Daftar Isi

1. [Arsitektur ML Ensemble](#1-arsitektur-ml-ensemble)
2. [Labeling Strategy](#2-labeling-strategy)
3. [Feature Engineering](#3-feature-engineering)
4. [Model 1 — XGBoost](#4-model-1--xgboost)
5. [Model 2 — LSTM](#5-model-2--lstm)
6. [Model 3 — CNN 1D](#6-model-3--cnn-1d)
7. [Model 4 — Meta-Learner](#7-model-4--meta-learner)
8. [Training Pipeline](#8-training-pipeline)
9. [Walk-Forward Validation](#9-walk-forward-validation)
10. [Regime-Aware Training](#10-regime-aware-training)
11. [Inference Pipeline](#11-inference-pipeline)
12. [Retraining Strategy](#12-retraining-strategy)
13. [Evaluasi & Metrics](#13-evaluasi--metrics)
14. [Struktur File](#14-struktur-file)

---

## 1. Arsitektur ML Ensemble

### Gambaran Umum

```
INPUT DATA (50+ fitur × multi-TF)
            │
    ┌───────┼───────────────┐
    ▼       ▼               ▼
┌────────┐ ┌──────────┐ ┌─────────┐
│XGBoost │ │  LSTM    │ │ CNN 1D  │
│        │ │          │ │         │
│Analisis│ │Sequence  │ │ Chart   │
│tabular │ │momentum  │ │ pattern │
│50+fitur│ │60 candle │ │ detect  │
└───┬────┘ └────┬─────┘ └────┬────┘
    │           │             │
    │  xgb_score│  lstm_score │  cnn_score
    │  win_prob │  trend_prob │  pattern_score
    └───────────┴─────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Meta-Learner    │
         │  (XGBoost ringan)│
         │                  │
         │ Gabungkan semua  │
         │ output + context │
         └────────┬─────────┘
                  │
                  ▼
         entry_quality_score (0.0–1.0)
         direction: LONG / SHORT / NONE
         sizing_mode: SMALL / MEDIUM / LARGE
```

### Peran Setiap Model

| Model | Input | Output | Keunggulan |
|-------|-------|--------|------------|
| XGBoost (3 variant) | 50+ fitur tabular | win_probability per variant | Pattern dari kombinasi indikator |
| LSTM | Sequence 60 candle × fitur | trend_continuation_prob | Memahami momentum yang berkembang |
| CNN 1D | Matrix candle (60 × fitur) | pattern_label + pattern_score | Mendeteksi chart pattern otomatis |
| Meta-Learner | Output 3 model + context | entry_quality_score final | Menggabungkan semua perspektif |

---

## 2. Labeling Strategy

### Prinsip: Forward-Looking Labeling

Bukan memprediksi "apakah harga naik?" tapi **"apakah trade ini profit dengan SL/TP kita?"**

```
Untuk setiap candle historis, simulasikan trade:

  Entry LONG  di close[i]
  SL          = close[i] × (1 - sl_pct)
  TP          = close[i] × (1 + tp_pct)

  Scan candle i+1, i+2, ..., i+MAX_HOLD:
    Jika low[j]  <= SL → LOSS (label = 0)
    Jika high[j] >= TP → WIN  (label = 1)
    Jika tidak ada dalam MAX_HOLD candle → NEUTRAL (exclude)

  Lakukan hal yang sama untuk SHORT:
    SL = close[i] × (1 + sl_pct)
    TP = close[i] × (1 - tp_pct)
```

### Parameter Labeling per Mode

| Mode | SL % | TP % | MAX_HOLD (candle 5M) | Keterangan |
|------|------|------|----------------------|------------|
| Opportunistic | 0.12% | 0.20% | 24 (2 jam) | Scalp cepat |
| Aligned | 0.15% | 0.40% | 72 (6 jam) | Mid-term |
| Conviction | 0.15% | 0.80% | 288 (24 jam) | Swing |

### Implementasi Python

```python
import pandas as pd
import numpy as np

def label_trades(df: pd.DataFrame, sl_pct: float, tp_pct: float,
                 max_hold: int, direction: str = "LONG") -> pd.Series:
    """
    df harus memiliki kolom: open, high, low, close
    Returns: Series dengan label 0 (LOSS), 1 (WIN), -1 (NEUTRAL/exclude)
    """
    labels = []
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    for i in range(len(df) - max_hold):
        entry = closes[i]

        if direction == "LONG":
            sl = entry * (1 - sl_pct)
            tp = entry * (1 + tp_pct)
            label = -1
            for j in range(i + 1, i + max_hold + 1):
                if lows[j] <= sl:
                    label = 0   # LOSS
                    break
                if highs[j] >= tp:
                    label = 1   # WIN
                    break
        else:  # SHORT
            sl = entry * (1 + sl_pct)
            tp = entry * (1 - tp_pct)
            label = -1
            for j in range(i + 1, i + max_hold + 1):
                if highs[j] >= sl:
                    label = 0
                    break
                if lows[j] <= tp:
                    label = 1
                    break

        labels.append(label)

    # Padding untuk candle terakhir yang tidak bisa dilabel
    labels.extend([-1] * max_hold)
    return pd.Series(labels, index=df.index)


def create_labeled_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Buat dataset dengan label untuk semua mode"""
    # Opportunistic
    df["label_long_opp"]  = label_trades(df, 0.0012, 0.0020, 24,  "LONG")
    df["label_short_opp"] = label_trades(df, 0.0012, 0.0020, 24,  "SHORT")

    # Aligned
    df["label_long_aln"]  = label_trades(df, 0.0015, 0.0040, 72,  "LONG")
    df["label_short_aln"] = label_trades(df, 0.0015, 0.0040, 72,  "SHORT")

    # Conviction
    df["label_long_con"]  = label_trades(df, 0.0015, 0.0080, 288, "LONG")
    df["label_short_con"] = label_trades(df, 0.0015, 0.0080, 288, "SHORT")

    # Filter: hanya pakai baris yang punya label valid (0 atau 1)
    return df[df["label_long_opp"] != -1]
```

### Distribusi Label yang Sehat

```
Target distribusi per label:
  WIN  (1): 40–60%
  LOSS (0): 40–60%

Jika WIN < 35% → SL terlalu ketat atau TP terlalu jauh
Jika WIN > 65% → Ada data leakage (bug serius, harus fix)

Cara handle imbalance (jika perlu):
  scale_pos_weight = count(LOSS) / count(WIN)
  → Parameter di XGBoost
```

---

## 3. Feature Engineering

### Prinsip

```
Setiap fitur harus menjawab satu pertanyaan spesifik:
  "Apa yang fitur ini tambahkan yang tidak ada di fitur lain?"

Hindari:
  - EMA20 dan EMA21 (hampir identik = multicollinearity)
  - Raw price (harga absolut tidak bermakna lintas waktu)
  - Fitur yang "melihat ke depan" (data leakage)

Gunakan:
  - Fitur yang dinormalisasi / relative (%, ratio)
  - Fitur yang sudah terbukti relevan di research quant
  - Kombinasi fitur yang menciptakan informasi baru
```

### 3.1 Fitur Harga & Trend (dari Linear Regression)

```python
# Per timeframe: 5M, 15M, 1H, 4H, Daily
# Total: 5 TF × 6 fitur = 30 fitur

price_trend_features = {
    # Linear Regression
    f"lr_slope_{tf}":         slope β₁ (dinormalisasi per ATR),
    f"lr_r2_{tf}":            R² goodness of fit,
    f"lr_residual_{tf}":      % deviasi harga dari midline,
    f"lr_channel_pos_{tf}":   posisi dalam channel (0=lower, 1=upper),

    # Trend Score
    f"trend_score_{tf}":      composite trend score (-1 hingga +1),
    f"roc_{tf}":              Rate of Change 10 periode (%),
}
```

### 3.2 Fitur EMA

```python
# Per timeframe: 5M, 15M, 1H, 4H, Daily
# Total: 5 TF × 5 fitur = 25 fitur

ema_features = {
    f"ema_align_score_{tf}":  score alignment EMA 20/50/200 (0–3),
    f"price_vs_ema20_{tf}":   % harga vs EMA20,
    f"price_vs_ema50_{tf}":   % harga vs EMA50,
    f"price_vs_ema200_{tf}":  % harga vs EMA200,
    f"ema20_slope_{tf}":      slope EMA20 (% per candle),
}
```

### 3.3 Fitur Oscillator

```python
# Per timeframe: 5M, 15M, 1H, 4H
# Total: ~4 TF × 8 fitur = 32 fitur

oscillator_features = {
    f"rsi_{tf}":              RSI 14,
    f"rsi_slope_{tf}":        RSI slope 5 periode,
    f"rsi_divergence_{tf}":   1=bullish div, -1=bearish, 0=none,
    f"macd_hist_{tf}":        MACD histogram (dinormalisasi per ATR),
    f"macd_cross_{tf}":       1=bullish cross, -1=bearish, 0=none,
}
```

### 3.4 Fitur Volatility

```python
# Per timeframe: 5M, 15M, 1H, 4H
# Total: ~4 TF × 5 fitur = 20 fitur

volatility_features = {
    f"atr_pct_{tf}":          ATR sebagai % harga,
    f"bb_width_{tf}":         Bollinger Band width (%),
    f"bb_position_{tf}":      %B posisi dalam band (0–1),
    f"bb_squeeze_{tf}":       1 jika squeeze terdeteksi,
    f"volatility_regime_{tf}":0=low, 1=normal, 2=high,
}
```

### 3.5 Fitur Fibonacci

```python
# Dari swing Daily dan 4H
# Total: 8 fitur

fibonacci_features = {
    "fib_zone_daily":         0=premium, 1=equil, 2=discount,
    "fib_distance_daily":     % jarak ke level fib terdekat,
    "in_golden_pocket":       1 jika harga di 61.8–65% zone,
    "fib_cluster_score":      0–3 berapa swing yang overlap,
    "fib_zone_4h":            0=premium, 1=equil, 2=discount,
    "fib_distance_4h":        % jarak ke level fib terdekat 4H,
    "nearest_fib_support":    % jarak ke fib support terdekat,
    "nearest_fib_resistance": % jarak ke fib resistance terdekat,
}
```

### 3.6 Fitur Struktur & Pattern

```python
# Total: 12 fitur

structure_features = {
    "market_structure":       0=downtrend, 1=ranging, 2=uptrend,
    "bos_detected":           Break of Structure terdeteksi (bool),
    "swing_range_pct":        % range antara last swing H dan L,
    "price_vs_swing_high":    % jarak ke swing high terakhir,
    "price_vs_swing_low":     % jarak ke swing low terakhir,
    "in_bull_ob":             harga dalam bullish order block (bool),
    "in_bear_ob":             harga dalam bearish order block (bool),
    "ob_distance_bull":       % jarak ke OB bullish terdekat,
    "candle_pattern":         0=bearish, 1=neutral, 2=bullish,
    "candle_strength":        0=weak, 1=moderate, 2=strong,
    "price_vs_vwap":          % jarak ke VWAP,
    "price_vs_pivot_pp":      % jarak ke Pivot Point,
}
```

### 3.7 Fitur Volume & Orderflow

```python
# Total: 10 fitur

orderflow_features = {
    "volume_ratio_5m":        rasio volume vs MA20,
    "volume_spike":           1 jika > 2x MA20 (bool),
    "cvd_bias":               -1=selling, 0=neutral, 1=buying,
    "funding_rate":           nilai funding rate (float),
    "funding_extreme":        1 jika |funding| > 0.05% (bool),
    "oi_change_4h":           % perubahan OI dalam 4 jam,
    "ls_ratio":               Long/Short ratio,
    "bid_ask_imbalance":      rasio bid vs ask volume,
    "liq_target_above":       USD di liquidation target atas,
    "liq_target_below":       USD di liquidation target bawah,
}
```

### 3.8 Fitur Confluence & Regime

```python
# Total: 8 fitur

confluence_features = {
    "confluence_score":       Multi-TF confluence (-1 hingga +1),
    "tf_aligned_count":       berapa TF yang align (0–5),
    "market_regime":          0=ranging, 1=bull, 2=bear, 3=volatile,
    "volatility_regime_global": 0=low, 1=normal, 2=high,
    "macro_bias":             -1=short, 0=neutral, 1=long,
    "market_mode":            0=opportunistic, 1=aligned, 2=conviction,
    "bottom_confirmed":       1 jika DeepSeek konfirmasi bottom, 1=inside, 2=above,
}
```

### Ringkasan Total Fitur

| Kategori | Jumlah Fitur |
|----------|-------------|
| Harga & Trend (LR) | 30 |
| EMA | 25 |
| Oscillator | 32 |
| Volatility | 20 |
| Fibonacci | 8 |
| Struktur & Pattern | 12 |
| Volume & Orderflow | 10 |
| Confluence & Regime | 8 |
| **Total** | **~145 fitur** |

---

## 4. Model 1 — XGBoost

### Arsitektur

Tiga model XGBoost dengan risk appetite berbeda, masing-masing dilatih dengan parameter yang berbeda:

| Variant | Target | Threshold Entry | Karakteristik |
|---------|--------|----------------|---------------|
| Conservative | Hanya WIN yang sangat yakin | win_prob > 0.70 | Jarang entry, akurasi tinggi |
| Balanced | Keseimbangan frekuensi & akurasi | win_prob > 0.58 | Trade moderat |
| Aggressive | Lebih banyak entry | win_prob > 0.52 | Lebih sering, akurasi lebih rendah |

### Hyperparameter

```python
# Conservative
xgb_conservative_params = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "max_depth":        4,          # dangkal = less overfitting
    "min_child_weight": 30,         # perlu banyak sample per leaf
    "subsample":        0.7,
    "colsample_bytree": 0.6,
    "learning_rate":    0.03,
    "n_estimators":     500,
    "scale_pos_weight": 1.2,        # sedikit penalti untuk false positive
    "reg_alpha":        0.1,        # L1 regularization
    "reg_lambda":       1.0,        # L2 regularization
    "random_state":     42,
}

# Balanced
xgb_balanced_params = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "max_depth":        5,
    "min_child_weight": 20,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "learning_rate":    0.05,
    "n_estimators":     400,
    "scale_pos_weight": 1.0,
    "reg_alpha":        0.05,
    "reg_lambda":       1.0,
    "random_state":     42,
}

# Aggressive
xgb_aggressive_params = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "max_depth":        6,
    "min_child_weight": 15,
    "subsample":        0.85,
    "colsample_bytree": 0.8,
    "learning_rate":    0.07,
    "n_estimators":     300,
    "scale_pos_weight": 0.9,
    "reg_alpha":        0.01,
    "reg_lambda":       0.5,
    "random_state":     42,
}
```

### Training

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

def train_xgboost(X_train, y_train, X_val, y_val,
                  params: dict, model_name: str) -> xgb.XGBClassifier:

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print(f"\n{model_name} — Top 15 Features:")
    print(importance.head(15))

    model.save_model(f"models/{model_name}.json")
    return model
```

### Output XGBoost

```python
def xgboost_inference(models: dict, features: pd.DataFrame) -> dict:
    prob_conservative = models["conservative"].predict_proba(features)[0][1]
    prob_balanced     = models["balanced"].predict_proba(features)[0][1]
    prob_aggressive   = models["aggressive"].predict_proba(features)[0][1]

    # Ensemble XGBoost (rata-rata berbobot)
    xgb_ensemble = (
        prob_conservative * 0.25 +
        prob_balanced     * 0.45 +
        prob_aggressive   * 0.30
    )

    return {
        "xgb_conservative": round(prob_conservative, 4),
        "xgb_balanced":     round(prob_balanced, 4),
        "xgb_aggressive":   round(prob_aggressive, 4),
        "xgb_ensemble":     round(xgb_ensemble, 4),
    }
```

---

## 5. Model 2 — LSTM

### Konsep

LSTM (Long Short-Term Memory) adalah Recurrent Neural Network yang mampu "mengingat" informasi dari candle-candle sebelumnya. Berbeda dari XGBoost yang melihat setiap candle secara independen, LSTM memahami **urutan dan momentum**.

```
Input: Sequence 60 candle terakhir
       Setiap candle: [close_norm, volume_norm, rsi, ema_align,
                       lr_slope, atr_pct, bb_position, macd_hist]
       Shape: (1, 60, 8) — batch=1, timesteps=60, features=8

Output: trend_continuation_probability (0.0–1.0)
        "Seberapa besar kemungkinan momentum ini berlanjut?"
```

### Arsitektur LSTM

```python
import torch
import torch.nn as nn

class TradingLSTM(nn.Module):
    def __init__(self, input_size: int = 8, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout,
            batch_first = True,
            bidirectional = False     # Unidirectional: tidak boleh lihat masa depan
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)         # Attention weight per timestep
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size)

        # Attention: fokus ke timestep yang paling penting
        attn_weights = self.attention(lstm_out)
        # attn_weights shape: (batch, seq_len, 1)

        context = (lstm_out * attn_weights).sum(dim=1)
        # context shape: (batch, hidden_size)

        output = self.classifier(context)
        return output.squeeze()
```

### Kenapa Ada Attention?

```
Tanpa Attention:
  LSTM baca 60 candle → output dari candle terakhir saja
  → Bisa "lupa" candle penting di tengah sequence

Dengan Attention:
  LSTM baca 60 candle → Attention putuskan candle mana
  yang paling relevan → Gabungkan secara berbobot

  Contoh: candle ke-45 ada volume spike besar
  → Attention kasih bobot tinggi ke candle itu
  → Model "ingat" event penting itu lebih kuat
```

### Feature Input LSTM

```python
LSTM_FEATURES = [
    "close_normalized",    # (close - mean) / std dalam window
    "volume_normalized",   # (volume - mean) / std
    "rsi_5m",             # RSI 14 5M (0–100, normalize ke 0–1)
    "ema_align_score_5m", # 0–3
    "lr_slope_5m",        # slope normalized
    "atr_pct_5m",         # ATR % (0–2)
    "bb_position_5m",     # %B (0–1)
    "macd_hist_5m",       # histogram normalized
]

SEQUENCE_LENGTH = 60      # 60 candle = 5 jam (5M candle)
```

### Training LSTM

```python
def train_lstm(model, train_loader, val_loader,
               epochs: int = 100, lr: float = 0.001) -> nn.Module:

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs  = model(X_val)
                val_loss = criterion(outputs, y_val.float())
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss   = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/lstm_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping di epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train={np.mean(train_losses):.4f}, "
                  f"val={avg_val_loss:.4f}")

    model.load_state_dict(torch.load("models/lstm_best.pt"))
    return model
```

---

## 6. Model 3 — CNN 1D

### Konsep

CNN (Convolutional Neural Network) 1D mendeteksi **pola lokal** dalam sequence harga — seperti Double Bottom, Head & Shoulders, Bullish Engulfing — secara otomatis tanpa perlu mendefinisikan pattern secara manual.

```
Analogi:
  CNN untuk gambar: filter 3×3 geser di gambar → deteksi tepi, bentuk
  CNN 1D untuk candle: filter geser di sequence → deteksi pattern candle

Input : Matrix (60 candle × 12 fitur)
Output: pattern_score (0.0–1.0) — seberapa kuat pattern bullish/bearish
```

### Arsitektur CNN 1D

```python
class TradingCNN1D(nn.Module):
    def __init__(self, in_channels: int = 12, seq_len: int = 60):
        super().__init__()

        # Convolutional blocks — deteksi pattern di berbagai skala
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),          # Downsample: 60 → 30
            nn.Dropout(0.2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),          # Downsample: 30 → 15
            nn.Dropout(0.2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global Average Pooling → (batch, 256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, features, seq_len) — CNN expects channels first
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.classifier(x).squeeze()
```

### Kenapa 3 Ukuran Filter Berbeda?

```
kernel_size=3 (conv_block1):
  Mendeteksi pattern SANGAT LOKAL (3 candle)
  → Bullish/Bearish Engulfing, Doji, Hammer

kernel_size=5 (conv_block2):
  Mendeteksi pattern LOKAL-MENENGAH (5 candle)
  → Morning Star, Evening Star, Three Soldiers

kernel_size=7 (conv_block3):
  Mendeteksi pattern MENENGAH (7+ candle)
  → Double Bottom, Double Top, Cup & Handle awal
```

### Feature Input CNN

```python
CNN_FEATURES = [
    # OHLC normalized
    "open_normalized",
    "high_normalized",
    "low_normalized",
    "close_normalized",
    # Volume
    "volume_normalized",
    # Candle properties
    "body_size",           # |close - open| / ATR
    "upper_wick",          # (high - max(open,close)) / ATR
    "lower_wick",          # (min(open,close) - low) / ATR
    "candle_direction",    # 1=bullish, -1=bearish
    # Context
    "rsi_5m",
    "bb_position_5m",
    "volume_ratio",        # volume vs MA20
]

# Input shape untuk CNN: (batch, 12 features, 60 candle)
# Note: CNN expects (batch, channels, length) — transpose dari LSTM
```

---

## 7. Model 4 — Meta-Learner

### Konsep

Meta-Learner adalah XGBoost ringan yang **menggabungkan output dari ketiga model** menjadi satu keputusan akhir. Dia belajar: "dalam kondisi seperti apa XGBoost benar tapi LSTM salah? Atau sebaliknya?"

```
Input Meta-Learner:
  - xgb_conservative, xgb_balanced, xgb_aggressive
  - lstm_score
  - cnn_score
  - agreement_score    (seberapa setuju ketiga model)
  - confluence_score   (dari TeknikalAgent)
  - market_regime
  - volatility_regime
  - direction          (LONG atau SHORT)

Output:
  - entry_quality_score (0.0–1.0)
  - direction           (LONG / SHORT / NONE)
  - sizing_mode         (SMALL / MEDIUM / LARGE)
```

### Training Meta-Learner

```python
def prepare_meta_features(xgb_preds: dict, lstm_preds: np.ndarray,
                           cnn_preds: np.ndarray,
                           context_features: pd.DataFrame) -> pd.DataFrame:
    meta_X = pd.DataFrame({
        "xgb_conservative": xgb_preds["conservative"],
        "xgb_balanced":     xgb_preds["balanced"],
        "xgb_aggressive":   xgb_preds["aggressive"],
        "xgb_ensemble":     xgb_preds["ensemble"],
        "lstm_score":       lstm_preds,
        "cnn_score":        cnn_preds,

        # Agreement score — seberapa setuju semua model
        "model_agreement":  np.std([
            xgb_preds["balanced"], lstm_preds, cnn_preds
        ], axis=0),

        # Context dari agent
        "confluence_score":      context_features["confluence_score"],
        "market_regime":         context_features["market_regime"],
        "volatility_regime":     context_features["volatility_regime"],
        "macro_bias":            context_features["macro_bias"],
        "market_mode":           context_features["market_mode"],
        "funding_extreme":       context_features["funding_extreme"],
        "in_golden_pocket":      context_features["in_golden_pocket"],
    })
    return meta_X


xgb_meta_params = {
    "objective":        "binary:logistic",
    "max_depth":        3,            # Sangat dangkal — meta learner harus simpel
    "min_child_weight": 10,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "learning_rate":    0.05,
    "n_estimators":     200,
    "random_state":     42,
}
```

### Output Final

```python
def meta_inference(meta_model, meta_features: pd.DataFrame,
                   final_decision: dict) -> dict:

    raw_score = meta_model.predict_proba(meta_features)[0][1]

    # Sesuaikan dengan market mode dari DeepSeek
    mode = final_decision.get("market_mode", "OPPORTUNISTIC")
    min_score = {
        "OPPORTUNISTIC": 0.52,
        "ALIGNED":       0.58,
        "CONVICTION":    0.65,
    }.get(mode, 0.55)

    # Tentukan sizing
    if raw_score >= 0.75:
        sizing = "LARGE"
    elif raw_score >= 0.60:
        sizing = "MEDIUM"
    else:
        sizing = "SMALL"

    entry_valid = raw_score >= min_score

    return {
        "entry_quality_score": round(raw_score, 4),
        "entry_valid":         entry_valid,
        "sizing_mode":         sizing if entry_valid else "NONE",
        "direction":           final_decision.get("overall_bias", "NEUTRAL"),
        "min_threshold":       min_score,
    }
```

---

## 8. Training Pipeline

### Urutan Training

```
1. Fetch data historis (365 hari OHLCV 5M dari Binance)
2. Hitung semua 145 fitur (dari INDICATORS.md)
3. Buat label forward-looking (WIN/LOSS)
4. Split: Train (70%) / Validation (15%) / Test (15%)
   PENTING: Split KRONOLOGIS, bukan random
5. Train XGBoost × 3
6. Train LSTM
7. Train CNN 1D
8. Kumpulkan prediksi ketiga model di validation set
9. Train Meta-Learner dari prediksi tersebut
10. Evaluasi di Test set (data yang belum pernah dilihat)
11. Simpan semua model
```

### Implementasi Pipeline

```python
def full_training_pipeline(symbol: str = "BTC/USDT:USDT",
                            days: int = 365) -> dict:
    print("=== TRAINING PIPELINE START ===")

    # Step 1: Data
    print("\n[1/9] Fetch data historis...")
    df_raw = fetch_historical_ohlcv(symbol, "5m", days=days)
    print(f"  Total candle: {len(df_raw):,}")

    # Step 2: Features
    print("\n[2/9] Hitung fitur...")
    df_feat = compute_all_features(df_raw)
    print(f"  Total fitur: {df_feat.shape[1]}")

    # Step 3: Labels
    print("\n[3/9] Buat label...")
    df_labeled = create_labeled_dataset(df_feat)
    win_rate = df_labeled["label_long_opp"].mean()
    print(f"  Win rate (opportunistic long): {win_rate:.1%}")

    # Step 4: Split kronologis
    print("\n[4/9] Split data...")
    n = len(df_labeled)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    df_train = df_labeled.iloc[:train_end]
    df_val   = df_labeled.iloc[train_end:val_end]
    df_test  = df_labeled.iloc[val_end:]
    print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

    # Step 5–7: Train base models
    print("\n[5/9] Train XGBoost...")
    xgb_models = train_all_xgboost(df_train, df_val)

    print("\n[6/9] Train LSTM...")
    lstm_model = train_lstm_model(df_train, df_val)

    print("\n[7/9] Train CNN 1D...")
    cnn_model = train_cnn_model(df_train, df_val)

    # Step 8: Kumpulkan prediksi di val set
    print("\n[8/9] Collect meta features...")
    meta_X, meta_y = collect_meta_predictions(
        xgb_models, lstm_model, cnn_model, df_val
    )

    # Step 9: Train Meta-Learner
    print("\n[9/9] Train Meta-Learner...")
    meta_model = train_xgboost(meta_X, meta_y,
                                meta_X, meta_y,  # kecil, ok pakai val=train
                                xgb_meta_params, "meta_learner")

    # Evaluasi di test set
    print("\n=== EVALUASI TEST SET ===")
    evaluate_all_models(xgb_models, lstm_model, cnn_model,
                         meta_model, df_test)

    return {
        "xgb_models":  xgb_models,
        "lstm_model":  lstm_model,
        "cnn_model":   cnn_model,
        "meta_model":  meta_model,
    }
```

---

## 9. Walk-Forward Validation

### Konsep

```
Validasi biasa (SALAH untuk trading):
  Random split → Data Feb ada di training, Jan ada di test
  → Model "tahu masa depan" secara tidak langsung (data leakage)

Walk-Forward (BENAR):
  Window 1: Train Jan–Jun | Test Jul
  Window 2: Train Jan–Jul | Test Aug
  Window 3: Train Jan–Aug | Test Sep
  ...
  Selalu test di data SETELAH training
```

### Implementasi

```python
def walk_forward_validation(df: pd.DataFrame,
                             train_months: int = 9,
                             test_months: int = 1,
                             n_splits: int = 6) -> dict:
    results = []
    candles_per_month = 30 * 24 * 12  # 5M candle per bulan

    for i in range(n_splits):
        train_start = 0
        train_end   = (train_months + i) * candles_per_month
        test_end    = train_end + test_months * candles_per_month

        if test_end > len(df):
            break

        df_train = df.iloc[train_start:train_end]
        df_test  = df.iloc[train_end:test_end]

        # Train model
        models = quick_train(df_train)

        # Evaluate
        metrics = evaluate(models, df_test)
        metrics["window"] = i + 1
        metrics["test_period"] = f"{df_test.index[0]} → {df_test.index[-1]}"
        results.append(metrics)

        print(f"Window {i+1}: WR={metrics['win_rate']:.1%} | "
              f"Sharpe={metrics['sharpe']:.2f} | "
              f"MaxDD={metrics['max_drawdown']:.1%}")

    avg_wr = np.mean([r["win_rate"] for r in results])
    print(f"\nAverage Win Rate: {avg_wr:.1%}")
    return {"windows": results, "avg_win_rate": avg_wr}
```

---

## 10. Regime-Aware Training

### Deteksi Market Regime

```python
def detect_market_regime(df: pd.DataFrame,
                          lookback: int = 100) -> pd.Series:
    """
    0 = RANGING
    1 = TRENDING BULL
    2 = TRENDING BEAR
    3 = HIGH VOLATILITY
    """
    regimes = []

    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]

        # ATR %
        atr_pct = window["atr_pct_1h"].mean()

        # Trend score
        lr_slope = window["lr_slope_4h"].iloc[-1]
        lr_r2    = window["lr_r2_4h"].iloc[-1]

        # Struktur
        structure = window["market_structure"].iloc[-1]

        if atr_pct > 0.8:
            regime = 3  # HIGH VOLATILITY
        elif lr_r2 > 0.6 and lr_slope > 0:
            regime = 1  # BULL TRENDING
        elif lr_r2 > 0.6 and lr_slope < 0:
            regime = 2  # BEAR TRENDING
        else:
            regime = 0  # RANGING

        regimes.append(regime)

    return pd.Series([None] * lookback + regimes, index=df.index)
```

### Strategi per Regime

```
RANGING (regime=0):
  - Strategi: buy support, sell resistance
  - XGBoost: fokus di fitur BB, RSI, Fibonacci level
  - Sizing: SMALL–MEDIUM

TRENDING BULL (regime=1):
  - Strategi: buy dips, TP lebar, hold lebih lama
  - XGBoost: fokus di EMA alignment, LR slope, momentum
  - Sizing: MEDIUM–LARGE

TRENDING BEAR (regime=2):
  - Strategi: sell rallies
  - XGBoost: sama seperti bull tapi untuk short
  - Sizing: MEDIUM

HIGH VOLATILITY (regime=3):
  - Strategi: sizing sangat kecil atau skip
  - Semua threshold dinaikkan 20%
  - Sizing: SMALL atau NONE
```

---

## 11. Inference Pipeline

### Alur saat Bot Berjalan Live

```python
def ml_inference(current_data: dict, final_decision: dict,
                 models: dict) -> dict:
    """
    Dipanggil setiap Fast Loop setelah ada cached final_decision
    dari DeepSeek yang belum expire
    """

    # 1. Hitung fitur real-time
    features_tabular = compute_realtime_features(current_data)

    # 2. XGBoost inference (< 1 detik)
    xgb_out = xgboost_inference(models["xgb"], features_tabular)

    # 3. LSTM inference (1–3 detik)
    sequence  = build_lstm_sequence(current_data, seq_len=60)
    lstm_out  = lstm_inference(models["lstm"], sequence)

    # 4. CNN inference (< 1 detik)
    matrix   = build_cnn_matrix(current_data, seq_len=60)
    cnn_out  = cnn_inference(models["cnn"], matrix)

    # 5. Meta-Learner
    meta_features = prepare_meta_features(
        xgb_out, lstm_out, cnn_out, features_tabular
    )
    final = meta_inference(models["meta"], meta_features, final_decision)

    return {
        "entry_quality_score": final["entry_quality_score"],
        "entry_valid":         final["entry_valid"],
        "direction":           final["direction"],
        "sizing_mode":         final["sizing_mode"],
        "detail": {
            "xgb": xgb_out,
            "lstm": lstm_out,
            "cnn": cnn_out,
        }
    }
```

---

## 12. Retraining Strategy

### Kapan Retrain?

| Trigger | Kondisi | Aksi |
|---------|---------|------|
| **Scheduled** | Setiap 30 hari | Full retrain semua model |
| **Performance** | Win rate < 45% dalam 50 trade terakhir | Retrain + alert |
| **Regime shift** | Regime berubah drastis | Retrain atau switch model |
| **Manual** | `python main.py --retrain` | Force retrain |

### Strategi Data

```
Saat retrain, gunakan data rolling window:
  - Selalu pakai 365 hari terakhir
  - Bukan akumulasi dari awal (data lama bisa tidak relevan)

Kecuali untuk fitur macro (struktur jangka panjang):
  - Data 2+ tahun untuk identifikasi support/resistance major
  - Hanya untuk kalkulasi Fibonacci macro dan LR daily
```

---

## 13. Evaluasi & Metrics

### Metrics yang Dipantau

```python
def evaluate_model(predictions, labels, trades_simulated) -> dict:
    return {
        # Klasifikasi
        "auc_roc":        roc_auc_score(labels, predictions),
        "precision":      precision_score(labels, predictions > 0.55),
        "recall":         recall_score(labels, predictions > 0.55),

        # Trading metrics
        "win_rate":       trades_simulated["win"].mean(),
        "avg_win_pct":    trades_simulated[trades_simulated["win"]==1]["pnl"].mean(),
        "avg_loss_pct":   trades_simulated[trades_simulated["win"]==0]["pnl"].mean(),
        "profit_factor":  (total_profit / total_loss) if total_loss != 0 else float("inf"),
        "expectancy":     win_rate * avg_win + loss_rate * avg_loss,
        "sharpe_ratio":   pnl_series.mean() / pnl_series.std() * np.sqrt(252),
        "max_drawdown":   compute_max_drawdown(equity_curve),
        "total_trades":   len(trades_simulated),
    }
```

### Target Metrics

| Metric | Minimum | Target |
|--------|---------|--------|
| Win Rate | > 50% | > 58% |
| Profit Factor | > 1.2 | > 1.5 |
| Sharpe Ratio | > 1.0 | > 1.5 |
| Max Drawdown | < 20% | < 10% |
| AUC-ROC | > 0.55 | > 0.62 |

---

## 14. Struktur File

```
models/
├── xgb_conservative.json
├── xgb_balanced.json
├── xgb_aggressive.json
├── lstm_best.pt
├── cnn_best.pt
├── meta_learner.json
└── training_metadata.json    ← tanggal training, metrics, params

core/
└── ml_engine/
    ├── __init__.py
    ├── feature_builder.py    ← compute_all_features()
    ├── labeler.py            ← forward-looking labeling
    ├── xgboost_model.py      ← train + inference XGBoost
    ├── lstm_model.py         ← TradingLSTM + training
    ├── cnn_model.py          ← TradingCNN1D + training
    ├── meta_learner.py       ← Meta-Learner
    ├── pipeline.py           ← full_training_pipeline()
    ├── inference.py          ← ml_inference() untuk live
    ├── evaluator.py          ← metrics + walk-forward
    └── regime_detector.py    ← detect_market_regime()
```

---

*MODELS.md v1.0 | AI Crypto Futures Trading Bot v4.2 | 2026-06-03*
*Dokumen berikutnya: CLAUDE.md — Entry Assistant specification & prompt engineering*