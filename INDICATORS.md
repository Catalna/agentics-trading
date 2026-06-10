# INDICATORS.md
# Spesifikasi Indikator Teknikal — AI Crypto Futures Trading Bot v4.2
> Dokumen ini mendefinisikan rumus matematis, cara kalkulasi Python, parameter,
> dan interpretasi semua indikator yang digunakan oleh agent dalam sistem.
> Dibaca bersama: `AGENTS.md`, `MODELS.md`

---

## Daftar Isi

1. [Linear Regression](#1-linear-regression)
2. [Trend Direction Detection](#2-trend-direction-detection)
3. [EMA & SMA](#3-ema--sma)
4. [RSI](#4-rsi-relative-strength-index)
5. [MACD](#5-macd)
6. [Bollinger Bands](#6-bollinger-bands)
7. [ATR](#7-atr-average-true-range)
8. [Fibonacci Retracement](#8-fibonacci-retracement)
9. [Fibonacci Extension](#9-fibonacci-extension)
10. [VWAP](#12-vwap-volume-weighted-average-price)
11. [Volume Analysis](#15-volume-analysis)
12. [Order Block Detection](#16-order-block-detection)
13. [Swing Structure](#17-swing-structure-hh-hl-lh-ll)
14. [Support & Resistance](#18-support--resistance)
15. [Multi-TF Confluence Score](#19-multi-tf-confluence-score)
16. [Volatility Regime](#20-volatility-regime)
17. [Library & Implementasi](#21-library--implementasi)

---

## 1. Linear Regression

### Konsep

Linear Regression adalah fondasi matematika untuk membaca arah dan kekuatan trend. Alih-alih melihat candle satu per satu, kita fitting sebuah garis lurus ke sekumpulan data harga dan mengukur karakteristiknya.

```
Model dasar:
  y = β₀ + β₁x + ε

Di mana:
  y  = harga (dependent variable)
  x  = waktu / index candle (independent variable)
  β₀ = intercept (nilai y saat x = 0)
  β₁ = slope (perubahan y per 1 unit x) ← INI yang paling penting
  ε  = error term (jarak aktual dari garis)
```

### Rumus Kalkulasi

```
Diberikan N candle dengan harga penutupan y₁, y₂, ..., yN
dan index waktu x₁, x₂, ..., xN (biasanya 0, 1, 2, ..., N-1)

Langkah 1 — Hitung mean:
  x̄ = (1/N) × Σxᵢ
  ȳ = (1/N) × Σyᵢ

Langkah 2 — Hitung slope β₁:
         Σ(xᵢ - x̄)(yᵢ - ȳ)
  β₁ = ─────────────────────
           Σ(xᵢ - x̄)²

Langkah 3 — Hitung intercept β₀:
  β₀ = ȳ - β₁ × x̄

Langkah 4 — Hitung nilai prediksi ŷ:
  ŷᵢ = β₀ + β₁ × xᵢ

Langkah 5 — Hitung R² (Goodness of Fit):
  SSE = Σ(yᵢ - ŷᵢ)²       ← Sum of Squared Errors
  SST = Σ(yᵢ - ȳ)²         ← Total Sum of Squares
  R² = 1 - (SSE / SST)

  Range R²: 0.0 – 1.0
  R² → 1.0 = harga bergerak sangat konsisten searah garis
  R² → 0.0 = harga choppy, tidak ada trend jelas
```

### Linear Regression Channel

```
Dari garis regresi, buat channel dengan standar deviasi residual:

  residuals = yᵢ - ŷᵢ  (untuk setiap candle)
  std_dev   = standar deviasi dari residuals

  Upper Channel = ŷᵢ + (multiplier × std_dev)   ← default: 2.0
  Lower Channel = ŷᵢ - (multiplier × std_dev)

Posisi harga dalam channel:
  channel_position = (harga - lower) / (upper - lower)
  Range: 0.0 (di lower) → 1.0 (di upper)
  0.5 = tepat di garis tengah

Residual harga saat ini:
  lr_residual_pct = (harga_sekarang - ŷ_sekarang) / harga_sekarang × 100
  Positif = harga di atas garis (premium)
  Negatif = harga di bawah garis (discount)
```

### Parameter per Agent

| Agent | Timeframe | Periode (candle) | Multiplier Channel |
|-------|-----------|-----------------|-------------------|
| MacroAgent | Daily | 200 | 2.0 |
| TrendAgent | 4H | 50 | 2.0 |
| TrendAgent | 1H | 50 | 1.5 |
| ScalpAgent | 5M | 20 | 1.5 |
| TeknikalAgent | Multi-TF | semua di atas | — |

### Implementasi Python

```python
import numpy as np

def linear_regression(prices: list[float], channel_mult: float = 2.0) -> dict:
    n = len(prices)
    x = np.arange(n, dtype=float)
    y = np.array(prices, dtype=float)

    x_mean = x.mean()
    y_mean = y.mean()

    # Slope dan intercept
    numerator   = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    beta1 = numerator / denominator          # slope
    beta0 = y_mean - beta1 * x_mean          # intercept

    # Predicted values
    y_hat = beta0 + beta1 * x

    # R-squared
    sse = np.sum((y - y_hat) ** 2)
    sst = np.sum((y - y_mean) ** 2)
    r2  = 1 - (sse / sst) if sst != 0 else 0

    # Residuals dan channel
    residuals  = y - y_hat
    std_dev    = residuals.std()
    upper      = y_hat[-1] + channel_mult * std_dev
    lower      = y_hat[-1] - channel_mult * std_dev
    midline    = y_hat[-1]

    current_price    = prices[-1]
    residual_pct     = (current_price - midline) / current_price * 100
    channel_position = (current_price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5

    return {
        "slope":            round(beta1, 6),
        "intercept":        round(beta0, 2),
        "r2":               round(r2, 4),
        "midline":          round(midline, 2),
        "upper":            round(upper, 2),
        "lower":            round(lower, 2),
        "residual_pct":     round(residual_pct, 4),
        "channel_position": round(channel_position, 4),  # 0=lower, 1=upper
        "trend_direction":  "UP" if beta1 > 0 else "DOWN" if beta1 < 0 else "FLAT",
    }
```

### Interpretasi

| Kondisi | Makna |
|---------|-------|
| `slope > 0` dan `r2 > 0.6` | Uptrend kuat dan konsisten |
| `slope > 0` dan `r2 < 0.3` | Bias naik tapi choppy |
| `slope ≈ 0` | Sideways / ranging |
| `slope < 0` dan `r2 > 0.6` | Downtrend kuat |
| `channel_position < 0.2` | Harga di area lower channel (discount) |
| `channel_position > 0.8` | Harga di area upper channel (premium) |
| `residual_pct < -1.5%` | Harga jauh di bawah garis, potensi mean reversion up |

---

## 2. Trend Direction Detection

### Konsep

Selain Linear Regression, ada beberapa cara matematis lain untuk mendeteksi apakah grafik sedang naik atau turun. Digunakan oleh semua agent sesuai timeframe-nya.

### 2.1 Slope of Moving Average

```
Mengukur apakah EMA sedang naik atau turun:

  ema_slope = (EMA[sekarang] - EMA[N candle lalu]) / N

  Positif = EMA sedang naik → uptrend
  Negatif = EMA sedang turun → downtrend

Normalisasi sebagai % per candle:
  ema_slope_pct = ema_slope / EMA[sekarang] × 100
```

### 2.2 Price Rate of Change (ROC)

```
Mengukur seberapa cepat harga berubah:

         harga[sekarang] - harga[N candle lalu]
  ROC = ─────────────────────────────────────── × 100
                  harga[N candle lalu]

Parameter: N = 10 (default)

Interpretasi:
  ROC > 0 = harga lebih tinggi dari N candle lalu (bullish)
  ROC < 0 = harga lebih rendah dari N candle lalu (bearish)
  |ROC| besar = pergerakan cepat / momentum kuat
```

### 2.3 Higher High / Lower Low (Trend Structure)

```
Algoritma deteksi otomatis:

1. Identifikasi swing points (local minima & maxima)
   Swing High: candle[i] high > candle[i-2..i+2] high semua
   Swing Low:  candle[i] low  < candle[i-2..i+2] low  semua

2. Bandingkan swing points berurutan:
   HH = swing_high[i] > swing_high[i-1]   (Higher High)
   HL = swing_low[i]  > swing_low[i-1]    (Higher Low)
   LH = swing_high[i] < swing_high[i-1]   (Lower High)
   LL = swing_low[i]  < swing_low[i-1]    (Lower Low)

3. Tentukan struktur:
   HH + HL = UPTREND
   LH + LL = DOWNTREND
   HH + LL atau LH + HL = RANGING / TRANSITION
```

### 2.4 Implementasi Python

```python
def detect_trend(closes: list[float], emas: dict, period_roc: int = 10) -> dict:
    prices = np.array(closes)

    # Rate of Change
    roc = (prices[-1] - prices[-period_roc]) / prices[-period_roc] * 100

    # EMA Slope (perubahan EMA20 dalam 5 candle terakhir)
    ema20 = emas.get("ema_20", [])
    ema_slope = (ema20[-1] - ema20[-5]) / 5 if len(ema20) >= 5 else 0
    ema_slope_pct = ema_slope / ema20[-1] * 100 if ema20[-1] != 0 else 0

    # Trend strength composite
    lr = linear_regression(closes[-50:])
    trend_score = (
        (1 if roc > 0 else -1) * min(abs(roc) / 2, 1) * 0.3 +
        (1 if ema_slope > 0 else -1) * min(abs(ema_slope_pct) * 10, 1) * 0.3 +
        (1 if lr["slope"] > 0 else -1) * lr["r2"] * 0.4
    )

    return {
        "roc":            round(roc, 4),
        "ema_slope_pct":  round(ema_slope_pct, 6),
        "lr_slope":       lr["slope"],
        "lr_r2":          lr["r2"],
        "trend_score":    round(trend_score, 4),   # -1.0 (strong down) to +1.0 (strong up)
        "trend_label":    "STRONG_UP" if trend_score > 0.5 else
                          "UP"        if trend_score > 0.1 else
                          "FLAT"      if abs(trend_score) <= 0.1 else
                          "DOWN"      if trend_score > -0.5 else "STRONG_DOWN",
    }
```

---

## 3. EMA & SMA

### Konsep

Moving Average menghaluskan pergerakan harga untuk menunjukkan arah trend.

### SMA (Simple Moving Average)

```
         y₁ + y₂ + ... + yN
  SMA  = ──────────────────
                N

Semua candle memiliki bobot yang sama.
```

### EMA (Exponential Moving Average)

```
EMA memberikan bobot lebih besar ke harga terbaru:

  Multiplier k = 2 / (N + 1)

  EMA[hari ini] = harga[hari ini] × k + EMA[kemarin] × (1 - k)

Contoh EMA 20:
  k = 2 / (20 + 1) = 0.0952
  EMA20 hari ini = harga × 0.0952 + EMA20 kemarin × 0.9048

EMA lebih responsif terhadap perubahan harga baru dibanding SMA.
```

### EMA Stack per Timeframe (Fibonacci-Based)

Semua periode EMA yang digunakan adalah angka Fibonacci murni, kecuali 200 yang dipertahankan karena self-fulfilling di kalangan institusi.

```
Deret Fibonacci: 1,1,2,3,5,8,13,21,34,55,89,144,233,377,...

Timeframe → EMA Set:
  Weekly / Daily  →  EMA 13 / EMA 144 / EMA 377
  4H              →  EMA 13 / EMA 34  / EMA 89
  1H              →  EMA 13 / EMA 34  / EMA 144
  15M             →  EMA 8  / EMA 21  / EMA 55
  5M              →  EMA 5  / EMA 13  / EMA 34

Rasio antar EMA mendekati Golden Ratio (φ ≈ 1.618):
  377 / 144 ≈ 2.618 = φ²
  144 / 89  ≈ 1.618 = φ
   89 / 55  ≈ 1.618 = φ
   55 / 34  ≈ 1.618 = φ
   34 / 21  ≈ 1.619 ≈ φ
   21 / 13  ≈ 1.615 ≈ φ
   13 / 8   ≈ 1.625 ≈ φ
```

### EMA Alignment Score

```
Untuk setiap timeframe, cek posisi ketiga EMA:

  score = 0
  if EMA_fast  > EMA_mid:  score += 1
  if EMA_mid   > EMA_slow: score += 1
  if harga     > EMA_fast: score += 1

  score = 3 → STRONG BULLISH (golden alignment)
  score = 2 → BULLISH
  score = 1 → MIXED
  score = 0 → BEARISH (death alignment)

Contoh 1H (EMA 13/34/144):
  EMA13 > EMA34 > EMA144 + harga > EMA13 → score = 3 → STRONG BULLISH

Multi-TF alignment (6 timeframe):
  total_score    = Σ score per timeframe (max = 18)
  alignment_pct  = total_score / 18 × 100
```

### Cross Signal

```
Golden Cross : EMA_fast cross EMA_mid dari bawah ke atas
               → Momentum bullish terkonfirmasi

Death Cross  : EMA_fast cross EMA_mid dari atas ke bawah
               → Momentum bearish terkonfirmasi

Power Cross  : EMA_fast cross EMA_slow (lebih jarang, lebih kuat)
               → Perubahan trend jangka panjang

Contoh di Daily (EMA 13/144/377):
  EMA13 cross EMA144 = signal medium-term (~5 bulan)
  EMA13 cross EMA377 = signal macro (~15 bulan) — sangat kuat
```

### Minimum Data yang Dibutuhkan

```
Weekly/Daily  → min 377 candle  (~1.5 tahun Daily, ~7 tahun Weekly)
4H            → min 89 candle   (~15 hari data 4H)
1H            → min 144 candle  (~6 hari data 1H)
15M           → min 55 candle   (~14 jam data 15M)
5M            → min 34 candle   (~3 jam data 5M)

Catatan Weekly:
  EMA 377 Weekly = 377 minggu = ~7.2 tahun
  Jika data tidak cukup, gunakan EMA 144 sebagai pengganti EMA 377
  hingga data tersedia
```

### Implementasi Python

```python
# EMA periods per timeframe
EMA_CONFIG = {
    "weekly":  {"fast": 13, "mid": 144, "slow": 377},
    "daily":   {"fast": 13, "mid": 144, "slow": 377},
    "4h":      {"fast": 13, "mid": 34,  "slow": 89},
    "1h":      {"fast": 13, "mid": 34,  "slow": 144},
    "15m":     {"fast": 8,  "mid": 21,  "slow": 55},
    "5m":      {"fast": 5,  "mid": 13,  "slow": 34},
}

def calculate_ema(prices: list[float], period: int) -> list[float]:
    """Hitung EMA dengan Exponential Smoothing."""
    k    = 2 / (period + 1)
    emas = [prices[0]]
    for price in prices[1:]:
        emas.append(price * k + emas[-1] * (1 - k))
    return emas

def ema_analysis(prices: list[float], timeframe: str) -> dict:
    """Hitung EMA lengkap untuk satu timeframe."""
    cfg   = EMA_CONFIG[timeframe]
    fast  = cfg["fast"]
    mid   = cfg["mid"]
    slow  = cfg["slow"]

    # Validasi data cukup
    if len(prices) < slow:
        raise ValueError(f"Data tidak cukup untuk {timeframe}: "
                         f"butuh {slow}, ada {len(prices)}")

    ema_fast_series = calculate_ema(prices, fast)
    ema_mid_series  = calculate_ema(prices, mid)
    ema_slow_series = calculate_ema(prices, slow)

    ema_fast = ema_fast_series[-1]
    ema_mid  = ema_mid_series[-1]
    ema_slow = ema_slow_series[-1]
    price    = prices[-1]

    # Alignment score
    score = sum([
        ema_fast > ema_mid,
        ema_mid  > ema_slow,
        price    > ema_fast,
    ])
    labels = {3: "STRONG_BULLISH", 2: "BULLISH", 1: "MIXED", 0: "BEARISH"}

    # Cross detection (fast vs mid)
    cross = "NONE"
    if len(ema_fast_series) >= 2 and len(ema_mid_series) >= 2:
        if ema_fast_series[-2] < ema_mid_series[-2] and ema_fast > ema_mid:
            cross = "GOLDEN_CROSS"
        elif ema_fast_series[-2] > ema_mid_series[-2] and ema_fast < ema_mid:
            cross = "DEATH_CROSS"

    # Power cross (fast vs slow)
    power_cross = "NONE"
    if len(ema_fast_series) >= 2 and len(ema_slow_series) >= 2:
        if ema_fast_series[-2] < ema_slow_series[-2] and ema_fast > ema_slow:
            power_cross = "BULLISH_POWER_CROSS"
        elif ema_fast_series[-2] > ema_slow_series[-2] and ema_fast < ema_slow:
            power_cross = "BEARISH_POWER_CROSS"

    # Slope EMA_fast (% per candle, 5 candle terakhir)
    ema_slope_pct = (
        (ema_fast_series[-1] - ema_fast_series[-5]) /
        ema_fast_series[-5] * 100
    ) if len(ema_fast_series) >= 5 else 0

    return {
        f"ema_{fast}":       round(ema_fast, 2),
        f"ema_{mid}":        round(ema_mid, 2),
        f"ema_{slow}":       round(ema_slow, 2),
        "alignment_score":   score,
        "alignment_label":   labels[score],
        "golden_cross":      cross == "GOLDEN_CROSS",
        "death_cross":       cross == "DEATH_CROSS",
        "power_cross":       power_cross,
        "ema_fast_slope_pct": round(ema_slope_pct, 6),
        "price_vs_fast_pct": round((price - ema_fast) / ema_fast * 100, 4),
        "price_vs_mid_pct":  round((price - ema_mid)  / ema_mid  * 100, 4),
        "price_vs_slow_pct": round((price - ema_slow) / ema_slow * 100, 4),
    }

def multi_tf_ema_confluence(tf_results: dict) -> dict:
    """Hitung confluence EMA dari semua timeframe."""
    scores = [r["alignment_score"] for r in tf_results.values()]
    total  = sum(scores)
    max_   = len(scores) * 3

    return {
        "total_alignment_score": total,
        "alignment_pct":         round(total / max_ * 100, 1),
        "all_bullish":           all(s >= 2 for s in scores),
        "all_bearish":           all(s <= 1 for s in scores),
        "tf_scores":             {tf: r["alignment_score"]
                                  for tf, r in tf_results.items()},
    }
```

### Interpretasi

| Kondisi | Makna |
|---------|-------|
| Score 3 semua TF | Extreme bullish confluence — Conviction Mode |
| Score 3 di Daily + 4H | Macro + mid-term aligned — Aligned Mode |
| Score 3 di 5M/15M saja | Local momentum, tapi macro belum confirm |
| Golden Cross Daily EMA13/144 | Signal medium-term sangat kuat |
| Power Cross Daily EMA13/377 | Signal macro — sangat jarang, sangat kuat |
| Harga di antara EMA34 dan EMA144 (1H) | Area konsolidasi, tunggu breakout |

---

## 4. RSI (Relative Strength Index)

### Konsep

RSI mengukur kecepatan dan perubahan pergerakan harga. Membandingkan rata-rata kenaikan vs rata-rata penurunan dalam N periode.

### Rumus

```
Langkah 1 — Hitung perubahan harga:
  Δ[i] = close[i] - close[i-1]

Langkah 2 — Pisahkan gain dan loss:
  gain[i] = Δ[i]  jika Δ[i] > 0, else 0
  loss[i] = |Δ[i]| jika Δ[i] < 0, else 0

Langkah 3 — Hitung rata-rata awal (periode pertama):
  avg_gain = mean(gain[1..N])
  avg_loss = mean(loss[1..N])

Langkah 4 — Periode berikutnya gunakan Wilder Smoothing:
  avg_gain = (avg_gain × (N-1) + gain[i]) / N
  avg_loss = (avg_loss × (N-1) + loss[i]) / N

Langkah 5 — Relative Strength dan RSI:
  RS  = avg_gain / avg_loss
  RSI = 100 - (100 / (1 + RS))

  Range: 0 – 100
  RSI > 70 = overbought (harga naik terlalu cepat)
  RSI < 30 = oversold (harga turun terlalu cepat)
  RSI 40-60 = zona netral, momentum sehat
```

### RSI Divergence

```
Bullish Divergence:
  Harga membuat Lower Low (LL)
  tapi RSI membuat Higher Low (HL)
  → Bearish momentum melemah, potensi reversal ke atas

Bearish Divergence:
  Harga membuat Higher High (HH)
  tapi RSI membuat Lower High (LH)
  → Bullish momentum melemah, potensi reversal ke bawah
```

### Implementasi Python

```python
def calculate_rsi(closes: list[float], period: int = 14) -> list[float]:
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    rsi_values = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs  = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(round(rsi, 4))

    return rsi_values

def rsi_analysis(rsi_values: list[float]) -> dict:
    current = rsi_values[-1]
    return {
        "rsi":        round(current, 2),
        "zone":       "OVERBOUGHT" if current > 70 else
                      "OVERSOLD"   if current < 30 else "NEUTRAL",
        "entry_valid": 35 <= current <= 65,   # zona ideal untuk entry
        "slope":      round(rsi_values[-1] - rsi_values[-5], 4),
    }
```

---

## 5. MACD

### Konsep

MACD mengukur perbedaan antara dua EMA untuk mendeteksi perubahan momentum.

### Rumus

```
MACD Line  = EMA(12) - EMA(26)
Signal Line = EMA(9) dari MACD Line
Histogram   = MACD Line - Signal Line

Interpretasi histogram:
  Histogram positif dan membesar → momentum bullish menguat
  Histogram positif tapi mengecil → momentum bullish melemah
  Histogram negatif dan membesar → momentum bearish menguat
  Histogram negatif tapi mengecil → momentum bearish melemah

Sinyal:
  MACD cross Signal dari bawah ke atas → BULLISH CROSS
  MACD cross Signal dari atas ke bawah → BEARISH CROSS
```

### MACD Divergence

```
Bullish MACD Divergence:
  Harga turun (LL) tapi Histogram membuat HL
  → Bearish momentum melemah

Bearish MACD Divergence:
  Harga naik (HH) tapi Histogram membuat LH
  → Bullish momentum melemah
```

### Implementasi Python

```python
def calculate_macd(closes: list[float],
                   fast: int = 12, slow: int = 26,
                   signal: int = 9) -> dict:
    ema_fast   = calculate_ema(closes, fast)
    ema_slow   = calculate_ema(closes, slow)
    macd_line  = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = calculate_ema(macd_line, signal)
    histogram  = [m - s for m, s in zip(macd_line, signal_line)]

    # Cross detection
    cross = "NONE"
    if len(histogram) >= 2:
        if histogram[-2] < 0 and histogram[-1] > 0:
            cross = "BULLISH_CROSS"
        elif histogram[-2] > 0 and histogram[-1] < 0:
            cross = "BEARISH_CROSS"

    return {
        "macd":        round(macd_line[-1], 4),
        "signal":      round(signal_line[-1], 4),
        "histogram":   round(histogram[-1], 4),
        "cross":       cross,
        "hist_trend":  "EXPANDING" if abs(histogram[-1]) > abs(histogram[-3]) else "CONTRACTING",
    }
```

---

## 6. Bollinger Bands

### Konsep

Bollinger Bands mengukur volatilitas dengan membuat channel di sekitar SMA berdasarkan standar deviasi.

### Rumus

```
Middle Band = SMA(20)
Upper Band  = SMA(20) + (2 × σ)
Lower Band  = SMA(20) - (2 × σ)

Di mana σ = standar deviasi harga dalam 20 periode:
         ┌─────────────────────────────┐
         │  Σ(yᵢ - ȳ)²               │
  σ =    │  ───────────               │
         │      N                     │
         └─────────────────────────────┘

BB Width = (Upper - Lower) / Middle × 100
  BB Width kecil = squeeze (volatility rendah, breakout imminent)
  BB Width besar = volatility tinggi

%B = (harga - Lower) / (Upper - Lower)
  %B = 1.0 → harga di Upper Band
  %B = 0.5 → harga di Middle Band
  %B = 0.0 → harga di Lower Band
  %B > 1.0 → harga di atas Upper (extremely overbought)
  %B < 0.0 → harga di bawah Lower (extremely oversold)
```

### Implementasi Python

```python
def bollinger_bands(closes: list[float], period: int = 20,
                    std_mult: float = 2.0) -> dict:
    prices = np.array(closes[-period:])
    middle = prices.mean()
    std    = prices.std()
    upper  = middle + std_mult * std
    lower  = middle - std_mult * std
    price  = closes[-1]

    bb_width    = (upper - lower) / middle * 100
    percent_b   = (price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5

    # BB Width percentile vs last 20 periods (squeeze detection)
    widths = []
    for i in range(20, len(closes)):
        p = np.array(closes[i-period:i])
        s = p.std()
        widths.append((p.max() - p.min()) / p.mean() * 100)
    width_percentile = (sum(w < bb_width for w in widths) / len(widths) * 100
                        if widths else 50)

    return {
        "upper":            round(upper, 2),
        "middle":           round(middle, 2),
        "lower":            round(lower, 2),
        "bb_width":         round(bb_width, 4),
        "percent_b":        round(percent_b, 4),
        "squeeze":          bb_width < np.percentile(widths, 20) if widths else False,
        "width_percentile": round(width_percentile, 1),
    }
```

---

## 7. ATR (Average True Range)

### Konsep

ATR mengukur volatilitas rata-rata pasar. Digunakan untuk menentukan jarak SL dan TP yang realistis.

### Rumus

```
True Range (TR) untuk setiap candle:
  TR = max(
    high - low,                    ← range candle hari ini
    |high - close_sebelumnya|,     ← gap up dari close kemarin
    |low  - close_sebelumnya|      ← gap down dari close kemarin
  )

ATR = Wilder Smoothing dari TR selama N periode:
  ATR[pertama] = mean(TR[1..N])
  ATR[i]       = (ATR[i-1] × (N-1) + TR[i]) / N

Default: N = 14

Penggunaan untuk SL:
  SL jarak = 1.0 × ATR   (normal)
  SL jarak = 1.5 × ATR   (volatile market)

SL Price (long):  entry - (multiplier × ATR)
SL Price (short): entry + (multiplier × ATR)
```

### Implementasi Python

```python
def calculate_atr(highs: list[float], lows: list[float],
                  closes: list[float], period: int = 14) -> dict:
    tr_values = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i]  - lows[i],
            abs(highs[i]  - closes[i-1]),
            abs(lows[i]   - closes[i-1])
        )
        tr_values.append(tr)

    # Wilder smoothing
    atr = sum(tr_values[:period]) / period
    for tr in tr_values[period:]:
        atr = (atr * (period - 1) + tr) / period

    price    = closes[-1]
    atr_pct  = atr / price * 100

    return {
        "atr":             round(atr, 2),
        "atr_pct":         round(atr_pct, 4),
        "sl_1x":           round(atr, 2),
        "sl_1_5x":         round(atr * 1.5, 2),
        "volatility":      "HIGH"   if atr_pct > 0.8 else
                           "LOW"    if atr_pct < 0.2 else "NORMAL",
    }
```

---

## 8. Fibonacci Retracement

### Konsep

Fibonacci Retracement mengidentifikasi area support/resistance potensial berdasarkan rasio Fibonacci — nisbah matematika yang ditemukan dalam pola alam dan diaplikasikan ke pasar finansial.

### Rasio Fibonacci

```
Deret Fibonacci: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ...

Rasio kunci (dari pembagian antar angka deret):
  0.0%   → titik awal swing (swing low untuk uptrend)
  23.6%  → retracement dangkal (23.6% dari swing)
  38.2%  → retracement pertama (Golden Ratio turunan)
  50.0%  → setengah swing (bukan Fibonacci murni tapi sangat kuat)
  61.8%  → Golden Ratio ← PALING PENTING (1/1.618 ≈ 0.618)
  78.6%  → retracement dalam (√0.618 ≈ 0.786)
  100.0% → titik akhir swing (swing high untuk uptrend)
```

### Rumus Kalkulasi

```
Untuk UPTREND (swing low → swing high):
  Retracement Level = swing_high - (ratio × (swing_high - swing_low))

  Contoh: swing_low = $58,000, swing_high = $73,500
    0.0%  → $73,500  (swing high)
   23.6%  → $73,500 - (0.236 × $15,500) = $69,842
   38.2%  → $73,500 - (0.382 × $15,500) = $67,581
   50.0%  → $73,500 - (0.500 × $15,500) = $65,750
   61.8%  → $73,500 - (0.618 × $15,500) = $63,921  ← Golden Pocket
   78.6%  → $73,500 - (0.786 × $15,500) = $61,317
  100.0%  → $58,000  (swing low)

Untuk DOWNTREND (swing high → swing low):
  Retracement Level = swing_low + (ratio × (swing_high - swing_low))
```

### Golden Pocket

```
Area paling kuat untuk reversal:
  Golden Pocket = zona antara 61.8% dan 65.0%
  (61.8% adalah Golden Ratio, 65% adalah zona validasi)

Ketika harga masuk ke Golden Pocket:
  + Konfirmasi lain (RSI oversold, volume spike, bullish candle)
  = Area entry dengan probabilitas tinggi
```

### Implementasi Python

```python
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_LABELS = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]

def fibonacci_retracement(swing_low: float, swing_high: float,
                           current_price: float,
                           trend: str = "UP") -> dict:
    swing_range = swing_high - swing_low
    levels = {}

    for ratio, label in zip(FIB_LEVELS, FIB_LABELS):
        if trend == "UP":
            levels[label] = round(swing_high - ratio * swing_range, 2)
        else:
            levels[label] = round(swing_low + ratio * swing_range, 2)

    # Cari level terdekat
    level_prices = list(levels.values())
    nearest_idx  = np.argmin([abs(current_price - lv) for lv in level_prices])
    nearest_label = FIB_LABELS[nearest_idx]
    nearest_price = level_prices[nearest_idx]
    distance_pct  = (current_price - nearest_price) / current_price * 100

    # Tentukan zone
    fib_ratio_current = (swing_high - current_price) / swing_range if trend == "UP" else \
                        (current_price - swing_low) / swing_range
    zone = ("PREMIUM"      if fib_ratio_current < 0.382 else
            "EQUILIBRIUM"  if fib_ratio_current < 0.618 else
            "DISCOUNT")

    # Golden Pocket
    golden_low  = levels["61.8%"]
    golden_high = swing_high - 0.65 * swing_range if trend == "UP" else \
                  swing_low  + 0.65 * swing_range
    in_golden_pocket = min(golden_low, golden_high) <= current_price <= max(golden_low, golden_high)

    # Support dan resistance levels
    if trend == "UP":
        support    = [lv for lv in sorted(level_prices) if lv < current_price]
        resistance = [lv for lv in sorted(level_prices) if lv > current_price]
    else:
        support    = [lv for lv in sorted(level_prices, reverse=True) if lv > current_price]
        resistance = [lv for lv in sorted(level_prices, reverse=True) if lv < current_price]

    return {
        "levels":             levels,
        "nearest_level":      nearest_label,
        "nearest_price":      nearest_price,
        "distance_pct":       round(distance_pct, 4),
        "zone":               zone,
        "in_golden_pocket":   in_golden_pocket,
        "nearest_support":    support[-1]    if support    else None,
        "nearest_resistance": resistance[0]  if resistance else None,
        "swing_low":          swing_low,
        "swing_high":         swing_high,
    }
```

---

## 9. Fibonacci Extension

### Konsep

Fibonacci Extension memproyeksikan target harga **melewati** swing high/low — digunakan untuk menentukan TP yang lebih ambisius berdasarkan momentum.

### Rasio Extension

```
Level extension umum:
  127.2%  → target pertama (√1.618 ≈ 1.272)
  161.8%  → target utama (Golden Ratio)
  200.0%  → target psikologis
  261.8%  → target jauh (1.618²)
```

### Rumus Kalkulasi

```
Membutuhkan 3 titik: A (swing start), B (swing end), C (retracement end)

Untuk projected move UP (setelah pullback):
  Extension Level = C + (ratio × |B - A|)

Contoh:
  A = $58,000 (swing low)
  B = $73,500 (swing high)
  C = $63,921 (pullback ke 61.8%)
  Swing range = $73,500 - $58,000 = $15,500

  127.2% extension → $63,921 + (1.272 × $15,500) = $83,617
  161.8% extension → $63,921 + (1.618 × $15,500) = $88,999
  200.0% extension → $63,921 + (2.000 × $15,500) = $94,921
```

### Implementasi Python

```python
EXT_LEVELS  = [1.0, 1.272, 1.618, 2.0, 2.618]
EXT_LABELS  = ["100%", "127.2%", "161.8%", "200%", "261.8%"]

def fibonacci_extension(point_a: float, point_b: float, point_c: float,
                         direction: str = "UP") -> dict:
    swing_range = abs(point_b - point_a)
    levels = {}

    for ratio, label in zip(EXT_LEVELS, EXT_LABELS):
        if direction == "UP":
            levels[label] = round(point_c + ratio * swing_range, 2)
        else:
            levels[label] = round(point_c - ratio * swing_range, 2)

    return {
        "levels":           levels,
        "point_a":          point_a,
        "point_b":          point_b,
        "point_c":          point_c,
        "swing_range":      round(swing_range, 2),
        "tp_conservative":  levels["127.2%"],
        "tp_primary":       levels["161.8%"],
        "tp_aggressive":    levels["200%"],
    }
```

### Penggunaan untuk TP

```
Conviction Mode (bottom confirmed):
  TP1 = Fibonacci Extension 127.2%  ← TP pertama, partial close
  TP2 = Fibonacci Extension 161.8%  ← TP utama
  TP3 = Fibonacci Extension 200.0%  ← hold sisanya dengan trailing stop

Scalping (Opportunistic Mode):
  Cukup gunakan Fibonacci Retracement level terdekat sebagai TP
  TP = nearest Fibonacci resistance di atas entry
```

---

## 10. VWAP (Volume Weighted Average Price)

### Konsep

VWAP adalah harga rata-rata yang sudah diperhitungkan terhadap volume. Digunakan institusi besar sebagai benchmark — harga di atas VWAP berarti pasar rela bayar lebih mahal dari rata-rata.

### Rumus

```
Typical Price (TP) per candle:
  TP = (High + Low + Close) / 3

VWAP kumulatif dari awal sesi:
         Σ(TPᵢ × Volumeᵢ)
  VWAP = ─────────────────
              ΣVolumeᵢ

Reset setiap awal sesi (daily VWAP reset jam 00:00 UTC)
```

### Implementasi Python

```python
def calculate_vwap(highs: list[float], lows: list[float],
                   closes: list[float], volumes: list[float]) -> dict:
    tp          = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    cum_tp_vol  = np.cumsum([t * v for t, v in zip(tp, volumes)])
    cum_vol     = np.cumsum(volumes)
    vwap_series = cum_tp_vol / cum_vol

    price        = closes[-1]
    vwap_current = vwap_series[-1]
    distance_pct = (price - vwap_current) / vwap_current * 100

    return {
        "vwap":          round(vwap_current, 2),
        "price_vs_vwap": "ABOVE" if price > vwap_current else "BELOW",
        "distance_pct":  round(distance_pct, 4),
        "vwap_support":  price > vwap_current,
    }
```

---

## 11. Volume Analysis

### Indikator Volume

```python
def volume_analysis(volumes: list[float], closes: list[float],
                    period_ma: int = 20) -> dict:
    vols      = np.array(volumes)
    vol_ma    = vols[-period_ma:].mean()
    vol_ratio = vols[-1] / vol_ma if vol_ma != 0 else 1.0

    # Volume Delta (proxy buy vs sell pressure)
    delta = closes[-1] - closes[-2]
    cvd   = sum(v if c > p else -v
                for v, c, p in zip(volumes[-20:], closes[-20:], closes[-21:-1]))

    return {
        "volume":         round(float(vols[-1]), 2),
        "volume_ma20":    round(float(vol_ma), 2),
        "volume_ratio":   round(float(vol_ratio), 4),
        "volume_spike":   vol_ratio > 2.0,
        "volume_trend":   "INCREASING" if vols[-1] > vols[-5:].mean() else "DECREASING",
        "cvd":            round(float(cvd), 2),
        "cvd_bias":       "BUYING" if cvd > 0 else "SELLING",
    }
```

---

## 12. Order Block Detection

### Konsep

Order Block adalah zona di mana institusi besar menempatkan order. Ditandai dengan candle besar yang sebelumnya menjadi titik awal pergerakan signifikan.

### Algoritma Deteksi

```python
def detect_order_blocks(opens, highs, lows, closes, volumes,
                        atr: float, lookback: int = 50) -> dict:
    blocks_bull, blocks_bear = [], []
    avg_vol = np.mean(volumes[-lookback:])

    for i in range(2, lookback):
        idx = -(lookback) + i
        body = abs(closes[idx] - opens[idx])

        # Candle besar (body > 1.5x ATR) dengan volume tinggi
        if body > 1.5 * atr and volumes[idx] > 1.3 * avg_vol:
            # Candle bullish → bearish OB (supply zone)
            if closes[idx] > opens[idx]:
                blocks_bear.append({
                    "high": highs[idx], "low": lows[idx],
                    "strength": body / atr
                })
            # Candle bearish → bullish OB (demand zone)
            else:
                blocks_bull.append({
                    "high": highs[idx], "low": lows[idx],
                    "strength": body / atr
                })

    price = closes[-1]

    def nearest(blocks, below=True):
        filtered = [b for b in blocks
                    if (b["low"] < price if below else b["high"] > price)]
        if not filtered:
            return None
        return max(filtered, key=lambda b: b["low"] if below else -b["high"])

    return {
        "ob_bullish":       blocks_bull,
        "ob_bearish":       blocks_bear,
        "nearest_bull_ob":  nearest(blocks_bull, below=True),
        "nearest_bear_ob":  nearest(blocks_bear, below=False),
        "price_in_bull_ob": any(b["low"] <= price <= b["high"] for b in blocks_bull),
        "price_in_bear_ob": any(b["low"] <= price <= b["high"] for b in blocks_bear),
    }
```

---

## 13. Swing Structure (HH, HL, LH, LL)

```python
def swing_structure(highs: list[float], lows: list[float],
                    closes: list[float], window: int = 5) -> dict:
    swing_highs, swing_lows = [], []

    for i in range(window, len(closes) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append((i, highs[i]))
        if lows[i]  == min(lows[i-window:i+window+1]):
            swing_lows.append((i, lows[i]))

    structure = "RANGING"
    bos       = False

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1][1] > swing_highs[-2][1]
        hl = swing_lows[-1][1]  > swing_lows[-2][1]
        lh = swing_highs[-1][1] < swing_highs[-2][1]
        ll = swing_lows[-1][1]  < swing_lows[-2][1]

        if hh and hl:
            structure = "UPTREND"
            bos       = hh
        elif lh and ll:
            structure = "DOWNTREND"
            bos       = ll
        else:
            structure = "RANGING"

    last_high = swing_highs[-1][1] if swing_highs else None
    last_low  = swing_lows[-1][1]  if swing_lows  else None

    return {
        "structure":   structure,
        "bos":         bos,
        "last_swing_high": last_high,
        "last_swing_low":  last_low,
        "swing_range": round(last_high - last_low, 2) if last_high and last_low else None,
    }
```

---

## 14. Support & Resistance

```python
def pivot_points(high: float, low: float, close: float) -> dict:
    pp = (high + low + close) / 3
    return {
        "pp": round(pp, 2),
        "r1": round(2 * pp - low, 2),
        "r2": round(pp + (high - low), 2),
        "s1": round(2 * pp - high, 2),
        "s2": round(pp - (high - low), 2),
    }
```

---

## 15. Multi-TF Confluence Score

```python
def confluence_score(tf_results: dict) -> dict:
    """
    tf_results = {
        "5m":    {"ema_score": 2, "rsi": 52, "lr_slope": 0.002},
        "15m":   {...},
        "1h":    {...},
        "4h":    {...},
        "daily": {...},
    }
    """
    scores = []
    for tf, data in tf_results.items():
        s = 0
        # EMA alignment (+1 bullish, -1 bearish)
        s += (data["ema_score"] - 1.5) / 1.5

        # RSI (40-60 = neutral, <40 = bearish, >60 = bullish)
        rsi = data.get("rsi", 50)
        s += 0.5 if 40 < rsi < 60 else (1 if rsi <= 40 else -1)

        # LR slope
        slope = data.get("lr_slope", 0)
        s += 1 if slope > 0 else (-1 if slope < 0 else 0)

        scores.append(s / 3)   # normalize per TF

    total = np.mean(scores)

    return {
        "confluence_score": round(float(total), 4),
        "label": ("STRONG_BULL" if total > 0.6 else
                  "BULL"        if total > 0.2 else
                  "NEUTRAL"     if abs(total) <= 0.2 else
                  "BEAR"        if total > -0.6 else "STRONG_BEAR"),
        "tf_count_aligned": sum(1 for s in scores if s > 0.2),
    }
```

---

## 16. Volatility Regime

```python
def volatility_regime(atr_pct: float, bb_width: float,
                      bb_width_percentile: float) -> dict:
    regime = ("HIGH"   if atr_pct > 0.8 or bb_width_percentile > 75 else
              "LOW"    if atr_pct < 0.2 or bb_width_percentile < 25 else
              "NORMAL")
    return {
        "regime":            regime,
        "atr_pct":           round(atr_pct, 4),
        "bb_width_pct":      round(bb_width_percentile, 1),
        "sl_multiplier":     1.5 if regime == "HIGH" else
                             0.8 if regime == "LOW"  else 1.0,
        "squeeze_imminent":  bb_width_percentile < 20,
    }
```

---

## 17. Library & Implementasi

### Stack yang Digunakan

```
Opsi C — Hybrid:
  Kalkulasi manual (numpy + pandas) : LR, Fibonacci, Swing Structure,
                                      Order Block, Confluence, Pivot
  Library TA                        : RSI, EMA, MACD, BB, ATR, Stoch,
                                      VWAP
```

### requirements.txt (bagian indikator)

```
numpy>=1.26.0
pandas>=2.1.0
pandas-ta>=0.3.14b
ta-lib>=0.4.28        # opsional, fallback ke pandas-ta
ccxt>=4.2.0
```

### Struktur File

```
core/
└── indicators/
    ├── __init__.py
    ├── linear_regression.py   ← LR Channel + Trend Detection
    ├── moving_averages.py     ← EMA, SMA, VWAP
    ├── oscillators.py         ← RSI, MACD
    ├── volatility.py          ← ATR, Bollinger Bands, Volatility Regime
    ├── fibonacci.py           ← Retracement + Extension
    ├── structure.py           ← Swing, Order Block, S/R, Pivot
    ├── volume.py              ← Volume Analysis, CVD
    └── confluence.py          ← Multi-TF Confluence Score
```

### Konvensi Kode

```python
# Semua fungsi indikator:
# - Input  : list[float] atau np.ndarray
# - Output : dict dengan key yang konsisten
# - Tidak ada side effects
# - Raise ValueError jika data tidak cukup

def validate_data(data: list, min_length: int, name: str):
    if len(data) < min_length:
        raise ValueError(f"{name} membutuhkan minimal {min_length} data points, "
                         f"diberikan {len(data)}")
```

---
