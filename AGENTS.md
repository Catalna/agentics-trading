# AGENTS.md
# Spesifikasi Agent — AI Crypto Futures Trading Bot v4.2
> Dokumen ini mendefinisikan seluruh agent dalam sistem, domain analisis, indikator yang digunakan, dan format report output.
> **Update v1.1:** EMA diperbarui ke stack Fibonacci murni per timeframe.
> Dibaca bersama: `PRD.md`, `INDICATORS.md`, `MODELS.md`

---

## Daftar Isi

1. [Hierarki Agent](#1-hierarki-agent)
2. [Format Report Standar](#2-format-report-standar)
3. [MacroAgent](#3-macroagent)
4. [TrendAgent](#4-trendagent)
5. [ScalpAgent](#5-scalpagent)
6. [TeknikalAgent](#6-teknikalagent)
7. [OrderflowAgent](#7-orderflowagent)
8. [Qwen — Technical Manager](#8-qwen--technical-manager)
9. [Mistral — Sentiment Manager](#9-mistral--sentiment-manager)
10. [SentimenAgent](#10-sentimenagent)
11. [DeepSeek R1 — Chief Supervisor](#11-deepseek-r1--chief-supervisor)
12. [ML Engine](#12-ml-engine)
13. [Claude API — Entry Assistant](#13-claude-api--entry-assistant)
14. [Data Flow Lengkap](#14-data-flow-lengkap)

---

## 1. Hierarki Agent

```
DataEngine ──────────────────────────────── SentimenEngine
     │                                            │
     ▼                                            ▼
┌─────────────────────────────┐    ┌──────────────────────────┐
│    Qwen 2.5:7b              │    │    Mistral:7b             │
│    Technical Manager        │    │    Sentiment Manager      │
│                             │    │                           │
│  ┌──────────────────────┐   │    │  ┌─────────────────────┐  │
│  │ MacroAgent           │   │    │  │ SentimenAgent       │  │
│  │ TrendAgent           │◄──┤    │  └─────────────────────┘  │
│  │ ScalpAgent           │   │    └────────────┬─────────────┘
│  │ TeknikalAgent        │   │                 │
│  │ OrderflowAgent       │   │                 │
│  └──────────┬───────────┘   │                 │
│  report A ▲▼ ke Qwen        │                 │
│  Qwen kompilasi             │                 │
└────────────┬────────────────┘                 │
             │ technical_report (Opsi A)         │ sentiment_report (Opsi A)
             └──────────────┬───────────────────┘
                            ▼
               ┌────────────────────────┐
               │   DeepSeek R1:7b       │
               │   Chief Supervisor     │
               └────────────┬───────────┘
                            │ final_decision
                            ▼
               ┌────────────────────────┐
               │   ML Engine            │
               │   XGBoost Ensemble     │
               └─────┬──────────────────┘
                     │ entry_proposal + detail_data (Opsi B)
               ┌─────┴──────┐
               ▼             ▼
         Executor       Claude API
         Standby        Entry Assistant
                        (opsional)
```

### Ringkasan Peran

| Agent / LLM | Lapor ke | Domain | Loop |
|-------------|----------|--------|------|
| MacroAgent | Qwen | Weekly/Daily — struktur besar | Slow |
| TrendAgent | Qwen | 4H/1H — momentum & arah | Slow |
| ScalpAgent | Qwen | 15M/5M — entry timing & momentum | Fast + Slow |
| TeknikalAgent | Qwen | Multi-TF — matematika & confluence | Slow |
| OrderflowAgent | Qwen | Real-time — funding, OI, orderbook | Slow + Trigger |
| SentimenAgent | Mistral | Real-time — berita, Fear&Greed | Slow |
| Qwen | DeepSeek | Kompilasi 5 agent teknikal | Slow |
| Mistral | DeepSeek | Kompilasi sentimen | Slow |
| DeepSeek R1 | ML Engine | Final decision, market mode | Slow |
| ML Engine | Executor + Claude | Entry proposal konkret | Fast |
| Claude API | Executor | Validasi entry (opsional) | Fast |

---

## 2. Format Report Standar

### Opsi A — Report Ringkas (Agent → Qwen, Qwen → DeepSeek)

Digunakan untuk komunikasi antar LLM. Ringkas agar tidak membebani context window.

```json
{
  "agent": "MacroAgent",
  "timestamp": "2026-06-03T14:32:00Z",
  "timeframe": "1D",
  "bias": "LONG",
  "confidence": 0.74,
  "market_condition": "BOTTOM_ZONE",
  "key_points": [
    "Harga di area $60K–$62K, zona bottom historis",
    "EMA200 daily di $58.2K — harga masih di atas",
    "Linear Regression slope positif lemah (+0.0012)"
  ],
  "risk_flags": [
    "Volume belum konfirmasi bounce"
  ]
}
```

### Opsi B — Report Detail (ML Engine & Claude)

Digunakan sebagai feature input untuk ML Engine dan konteks untuk Claude. Berisi semua nilai numerik indikator.

```json
{
  "agent": "MacroAgent",
  "timestamp": "2026-06-03T14:32:00Z",
  "timeframe": "1D",
  "bias": "LONG",
  "confidence": 0.74,
  "indicators": {
    "lr_slope": 0.0012,
    "lr_r2": 0.76,
    "lr_residual_pct": -0.8,
    "lr_channel_upper": 71200,
    "lr_channel_lower": 68400,
    "ema_20": 69850,
    "ema_50": 67200,
    "ema_200": 58200,
    "ema_alignment": "BULLISH",
    "rsi_14": 48.2,
    "fib_level": "0.618",
    "fib_zone": "DISCOUNT",
    "fib_distance_pct": 1.2,
    "atr_14": 1240,
    "volume_vs_ma20": 0.87,
    "swing_high": 73500,
    "swing_low": 58000
  },
  "key_points": ["..."],
  "risk_flags": ["..."]
}
```

---

## 3. MacroAgent

### Identitas

| Parameter | Nilai |
|-----------|-------|
| Nama | `MacroAgent` |
| Lapor ke | Qwen (Technical Manager) |
| Timeframe | Weekly, Daily |
| Tujuan | Menentukan struktur pasar besar, estimasi bottom/top, regime |
| Loop | Slow Loop (setiap 5–10 menit) |

### Tugas Utama

MacroAgent melihat "gambar besar" — di mana harga berada dalam konteks historis jangka panjang. Dia yang menentukan apakah kita sedang di bull market, bear market, atau ranging, dan apakah harga mendekati bottom atau top signifikan.

### Indikator yang Digunakan

#### Linear Regression Channel
```
Periode : 200 candle Daily
Output  :
  - lr_slope      : arah dan kekuatan trend jangka panjang
  - lr_r2         : kualitas trend (0 = choppy, 1 = perfect trend)
  - lr_upper      : batas atas channel (+2 std dev)
  - lr_lower      : batas bawah channel (-2 std dev)
  - lr_residual   : posisi harga relatif terhadap garis tengah (%)

Interpretasi:
  slope > 0 + R² > 0.6  → uptrend kuat
  slope ≈ 0             → sideways / ranging
  harga di lr_lower     → area discount / potential bottom
  harga di lr_upper     → area premium / potential top
```

#### EMA (Exponential Moving Average)
```
Periode : EMA 13 / EMA 144 / EMA 377 (Weekly & Daily)
Output  :
  - ema_13, ema_144, ema_377     : nilai absolut
  - ema_alignment                : STRONG_BULLISH / BULLISH / MIXED / BEARISH
    STRONG_BULLISH = EMA13 > EMA144 > EMA377 + harga > EMA13
    BULLISH        = EMA13 > EMA144 > EMA377
    BEARISH        = EMA13 < EMA144 < EMA377
  - golden_cross                 : EMA13 cross EMA144 dari bawah (bool)
  - power_cross                  : EMA13 cross EMA377 dari bawah (bool)
  - price_vs_ema13_pct           : % harga vs EMA13
  - price_vs_ema144_pct          : % harga vs EMA144
  - price_vs_ema377_pct          : % harga vs EMA377

Interpretasi:
  Harga > EMA377 Daily  → macro bull market sesungguhnya
  Harga < EMA377 Daily  → macro bear market
  Harga antara EMA144 dan EMA377 → transisi / konsolidasi panjang
  Power Cross EMA13/377 → perubahan regime sangat kuat (jarang terjadi)
```

#### Fibonacci Retracement
```
Input   : swing_high dan swing_low (otomatis dari 52-week high/low)
Level   : 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0
Output  :
  - fib_level_nearest : level fib terdekat dengan harga saat ini
  - fib_zone          : PREMIUM (>0.5) / EQUILIBRIUM (0.382-0.618) / DISCOUNT (<0.382)
  - fib_distance_pct  : % jarak harga dari level fib terdekat
  - fib_support[]     : array level support fib di bawah harga
  - fib_resistance[]  : array level resistance fib di atas harga

Interpretasi:
  0.618–0.786 dari swing high → area golden pocket (strong support)
  0.786–1.0                   → deep discount, potential reversal
```

#### Volume Analysis
```
Periode : Volume MA 20 Daily
Output  :
  - volume_vs_ma20  : rasio volume candle vs MA20 (>1 = above average)
  - volume_trend    : INCREASING / DECREASING / FLAT

Interpretasi:
  Bounce dari bottom + volume > 1.5x MA → konfirmasi kuat
  Bounce dari bottom + volume < 0.8x MA → perlu konfirmasi lebih lanjut
```

#### Swing Structure
```
Output  :
  - swing_high      : higher high terakhir (52 minggu)
  - swing_low       : lower low terakhir (52 minggu)
  - market_structure: UPTREND (HH+HL) / DOWNTREND (LH+LL) / RANGING
  - bos_detected    : Break of Structure terdeteksi? (bool)

Interpretasi:
  BOS bullish = harga tembus swing high sebelumnya → konfirmasi reversal
```

#### ATR (Average True Range)
```
Periode : ATR 14 Daily
Output  :
  - atr_14          : nilai ATR absolut (dalam USD)
  - atr_pct         : ATR sebagai % dari harga
  - volatility_regime: HIGH / NORMAL / LOW

Interpretasi:
  ATR tinggi → pasar volatile, perlebar SL
  ATR rendah → pasar tenang, TP lebih mudah tercapai
```

### Opsi A Output ke Qwen

```json
{
  "agent": "MacroAgent",
  "bias": "LONG | SHORT | NEUTRAL",
  "confidence": 0.0–1.0,
  "market_condition": "BOTTOM_ZONE | TOP_ZONE | UPTREND | DOWNTREND | RANGING",
  "bottom_confirmed": false,
  "key_points": ["max 4 poin ringkas"],
  "risk_flags": ["max 2 flag"]
}
```

---

## 4. TrendAgent

### Identitas

| Parameter | Nilai |
|-----------|-------|
| Nama | `TrendAgent` |
| Lapor ke | Qwen (Technical Manager) |
| Timeframe | 4H, 1H |
| Tujuan | Konfirmasi arah trend menengah, momentum, premium/discount zone |
| Loop | Slow Loop |

### Tugas Utama

TrendAgent adalah jembatan antara makro (harian) dan scalping (menit). Dia memastikan bahwa entry yang direncanakan searah dengan momentum menengah. Jika MacroAgent bilang "bottom di $60K" tapi TrendAgent bilang "4H masih downtrend", maka ini sinyal untuk tunggu dulu.

### Indikator yang Digunakan

#### Linear Regression (Mid-Term)
```
Periode : LR 50 candle 4H, LR 50 candle 1H
Output  :
  - lr_slope_4h, lr_slope_1h
  - lr_r2_4h, lr_r2_1h
  - lr_residual_4h, lr_residual_1h
  - lr_trend_alignment : apakah 4H dan 1H slope sama arah?

Interpretasi:
  lr_slope_4h > 0 dan lr_slope_1h > 0 → trend alignment bullish
  Harga di bawah LR midline → discount zone, favorable long
```

#### EMA Multi-TF
```
Periode : EMA 13 / 34 / 89 di 4H
          EMA 13 / 34 / 144 di 1H

Output  :
  - ema_alignment_4h   : STRONG_BULLISH / BULLISH / MIXED / BEARISH
  - ema_alignment_1h   : STRONG_BULLISH / BULLISH / MIXED / BEARISH
  - golden_cross_4h    : EMA13 cross EMA34 dari bawah di 4H (bool)
  - death_cross_4h     : EMA13 cross EMA34 dari atas di 4H (bool)
  - golden_cross_1h    : EMA13 cross EMA34 dari bawah di 1H (bool)
  - price_vs_ema13_4h  : % harga vs EMA13 di 4H
  - price_vs_ema34_4h  : % harga vs EMA34 di 4H
  - price_vs_ema89_4h  : % harga vs EMA89 di 4H
  - price_vs_ema13_1h  : % harga vs EMA13 di 1H
  - price_vs_ema34_1h  : % harga vs EMA34 di 1H
  - price_vs_ema144_1h : % harga vs EMA144 di 1H

Interpretasi:
  Golden cross EMA13/34 di 4H = momentum bullish menengah terkonfirmasi
  Harga bounce dari EMA34 4H  = area support dynamic, potential entry
  Harga di atas EMA144 1H     = bias intraday bullish jangka panjang
  EMA13 > EMA34 > EMA89 di 4H = triple Fibonacci alignment — sangat bullish
```

#### MACD (Moving Average Convergence Divergence)
```
Setting : Fast 12, Slow 26, Signal 9 — di 4H dan 1H
Output  :
  - macd_line_4h, signal_line_4h, histogram_4h
  - macd_cross_4h    : BULLISH_CROSS / BEARISH_CROSS / NONE
  - macd_divergence_4h: BULLISH / BEARISH / NONE
  - histogram_trend  : EXPANDING / CONTRACTING

Interpretasi:
  Bullish cross di 4H = momentum long terkonfirmasi
  Divergence bullish = potential reversal meskipun harga turun
```

#### RSI (Relative Strength Index)
```
Periode : RSI 14 di 4H dan 1H
Output  :
  - rsi_4h, rsi_1h
  - rsi_zone_4h      : OVERSOLD (<30) / NEUTRAL / OVERBOUGHT (>70)
  - rsi_divergence_4h: BULLISH / BEARISH / NONE
  - rsi_trend_4h     : apakah RSI membuat HH/HL atau LH/LL?

Interpretasi:
  RSI 4H di 40–60 + slope positif → momentum sehat
  RSI divergence bullish 4H → reversal signal kuat
```


#### Premium / Discount Zone
```
Kalkulasi berbasis LR Channel + Fibonacci 4H
Output  :
  - price_zone : PREMIUM / EQUILIBRIUM / DISCOUNT
  - zone_score : 0.0–1.0 (1.0 = deep discount, ideal untuk long)

Interpretasi:
  DISCOUNT + macro LONG = area favorit untuk long entry
  PREMIUM + macro SHORT = area favorit untuk short entry
```

### Opsi A Output ke Qwen

```json
{
  "agent": "TrendAgent",
  "bias": "LONG | SHORT | NEUTRAL",
  "confidence": 0.0–1.0,
  "trend_strength": "STRONG | MODERATE | WEAK",
  "price_zone": "PREMIUM | EQUILIBRIUM | DISCOUNT",
  "momentum_aligned": true,
  "key_points": ["max 4 poin"],
  "risk_flags": ["max 2 flag"]
}
```

---

## 5. ScalpAgent

### Identitas

| Parameter | Nilai |
|-----------|-------|
| Nama | `ScalpAgent` |
| Lapor ke | Qwen (Technical Manager) |
| Timeframe | 15M, 5M |
| Tujuan | Cari area entry timing presisi, momentum jangka pendek, scalping opportunities |
| Loop | Fast Loop (monitor) + Slow Loop (analisis penuh) |

### Tugas Utama

ScalpAgent adalah "ujung tombak" eksekusi. Dia tidak peduli dengan trend harian — tugasnya menemukan **momentum lokal** yang cukup kuat untuk entry scalping. Dia bekerja di timeframe kecil dan selalu aktif, baik saat macro confirmed maupun belum.

Dua mode operasi:
- **Opportunistic**: macro belum jelas → scalp bebas long/short berdasarkan momentum lokal
- **Aligned**: macro confirmed → hanya scalp searah macro, counter-trend dengan sizing kecil

### Indikator yang Digunakan

#### RSI + Divergence
```
Periode : RSI 14 di 15M dan 5M
Output  :
  - rsi_5m, rsi_15m
  - rsi_zone_5m      : OVERSOLD / NEUTRAL / OVERBOUGHT
  - rsi_divergence_5m: BULLISH / BEARISH / NONE
  - rsi_slope_5m     : apakah RSI naik atau turun?
  - rsi_entry_valid  : bool (RSI 5M di 35–65 = zona netral, ideal entry)

Interpretasi:
  RSI 5M 45–55 + slope naik → momentum long sehat, bagus entry
  RSI 5M > 70 → overbought, hindari long baru
  RSI divergence bullish 15M → reversal scalp opportunity
```

#### EMA Scalping
```
Periode : EMA 8 / 21 / 55 di 15M
          EMA 5 / 13 / 34 di 5M

Output  :
  - ema_alignment_15m  : STRONG_BULLISH / BULLISH / MIXED / BEARISH
  - ema_alignment_5m   : STRONG_BULLISH / BULLISH / MIXED / BEARISH
  - golden_cross_5m    : EMA5 cross EMA13 dari bawah (bool) — entry signal
  - death_cross_5m     : EMA5 cross EMA13 dari atas (bool) — short signal
  - golden_cross_15m   : EMA8 cross EMA21 dari bawah (bool)
  - price_vs_ema5_5m   : % harga vs EMA5
  - price_vs_ema13_5m  : % harga vs EMA13
  - price_vs_ema8_15m  : % harga vs EMA8
  - price_vs_ema21_15m : % harga vs EMA21

Interpretasi:
  Golden Cross EMA5/13 di 5M = entry scalp long paling cepat
  EMA5 > EMA13 > EMA34 di 5M = micro-trend bullish solid
  Harga pullback ke EMA13 5M + bounce = area entry ideal
  EMA8 > EMA21 > EMA55 di 15M = konfirmasi tambahan dari TF lebih besar
```

#### Bollinger Bands
```
Periode : BB(20, 2) di 15M dan 5M
Output  :
  - bb_squeeze_5m    : bool — volatility sangat rendah, breakout imminent
  - bb_position_5m   : %B posisi harga dalam band (0=lower, 0.5=mid, 1=upper)
  - bb_expansion_5m  : bool — band melebar, momentum aktif
  - bb_squeeze_15m   : bool
  - bb_position_15m  : %B

Interpretasi:
  bb_position < 0.2 + RSI oversold → scalp long opportunity
  bb_squeeze = True → tunggu breakout arah sebelum entry
  bb_expansion = True → momentum sedang kuat, entry lebih aman
```

#### Order Block Detection
```
Definisi : Candle besar (body > 1.5x ATR) yang sebelumnya menyebabkan
           pergerakan signifikan, kini menjadi zona supply/demand
Output  :
  - ob_bullish[]     : array zona order block bullish (demand zone)
  - ob_bearish[]     : array zona order block bearish (supply zone)
  - ob_nearest_bull  : OB bullish terdekat di bawah harga
  - ob_nearest_bear  : OB bearish terdekat di atas harga
  - ob_distance_pct  : % jarak harga dari OB terdekat

Interpretasi:
  Harga masuk ke OB bullish = area demand kuat, scalp long
  Harga masuk ke OB bearish = area supply kuat, scalp short
```

#### Candle Pattern Recognition
```
Pattern yang dideteksi:
  Bullish: Hammer, Bullish Engulfing, Morning Star, Piercing Line,
           Bullish Harami, Three White Soldiers
  Bearish: Shooting Star, Bearish Engulfing, Evening Star,
           Dark Cloud Cover, Bearish Harami, Three Black Crows
  Neutral: Doji (ketidakpastian)

Output  :
  - candle_pattern_5m  : nama pattern atau NONE
  - candle_bias_5m     : BULLISH / BEARISH / NEUTRAL
  - candle_strength_5m : STRONG / MODERATE / WEAK

Interpretasi:
  Bullish Engulfing di OB bullish = konfirmasi ganda untuk long
```

#### Volume + Volume Delta
```
Output  :
  - volume_5m          : volume candle terakhir
  - volume_vs_ma20_5m  : rasio vs MA20
  - volume_spike_5m    : bool (volume > 2x MA20)
  - delta_5m           : buy volume - sell volume (proxy dari bid/ask)
  - cvd_5m             : Cumulative Volume Delta (trend buying vs selling pressure)

Interpretasi:
  Volume spike + arah up = konfirmasi momentum long
  CVD naik tapi harga turun = bullish divergence (smart money buying)
```

#### Linear Regression (Short-Term)
```
Periode : LR 20 candle 5M
Output  :
  - lr_slope_5m      : arah micro-trend
  - lr_r2_5m         : kualitas micro-trend
  - lr_residual_5m   : posisi harga vs garis regresi (% overshoot)

Interpretasi:
  lr_slope > 0 + r2 > 0.5 → micro-uptrend solid
  lr_residual tinggi       → harga terlalu jauh dari mean, potensi pullback
```

#### ATR (Scalping Context)
```
Periode : ATR 14 di 5M dan 15M
Output  :
  - atr_5m, atr_15m
  - atr_ratio          : atr_5m / atr_15m (volatility context)
  - min_tp_pct         : ATR-based minimum TP yang worth it
  - recommended_sl_pct : 1x ATR 5M sebagai SL reference

Interpretasi:
  ATR kecil = pasar tenang, TP lebih mudah tercapai
  ATR besar = perlebar SL, jangan paksa entry
```

### Opsi A Output ke Qwen

```json
{
  "agent": "ScalpAgent",
  "bias": "LONG | SHORT | NEUTRAL",
  "confidence": 0.0–1.0,
  "entry_timing": "NOW | WAIT | AVOID",
  "momentum_strength": "STRONG | MODERATE | WEAK",
  "nearest_ob": { "type": "BULLISH", "zone": [69800, 69950] },
  "key_points": ["max 4 poin"],
  "risk_flags": ["max 2 flag"]
}
```

---

## 6. TeknikalAgent

### Identitas

| Parameter | Nilai |
|-----------|-------|
| Nama | `TeknikalAgent` |
| Lapor ke | Qwen (Technical Manager) |
| Timeframe | Multi-TF (semua timeframe) |
| Tujuan | Kalkulasi matematis lanjutan, confluence score, Fibonacci multi-TF |
| Loop | Slow Loop |

### Tugas Utama

TeknikalAgent adalah "matematikawan" sistem. Dia tidak fokus pada satu timeframe tapi menghitung **confluence** — seberapa banyak indikator dari berbagai timeframe menunjuk ke arah yang sama. Semakin tinggi confluence, semakin kuat sinyal.

### Indikator yang Digunakan

#### Multi-TF Confluence Score
```
Kalkulasi menggunakan EMA Fibonacci per timeframe:

  EMA Config per TF:
    Weekly/Daily : EMA 13 / 144 / 377
    4H           : EMA 13 / 34  / 89
    1H           : EMA 13 / 34  / 144
    15M          : EMA 8  / 21  / 55
    5M           : EMA 5  / 13  / 34

  Untuk setiap timeframe (Weekly, Daily, 4H, 1H, 15M, 5M):
    - EMA alignment score : +1 bullish (fast>mid>slow), -1 bearish, 0 mixed
    - RSI score           : +1 jika 40–60 dan naik, -1 jika >70 atau <30
    - LR slope score      : +1 jika positif + R²>0.5, -1 jika negatif

  confluence_score = sum(semua score) / max_possible_score
  Range: -1.0 (full bearish) hingga +1.0 (full bullish)

Output  :
  - confluence_score    : -1.0 hingga +1.0
  - confluence_label    : STRONG_BULL / BULL / NEUTRAL / BEAR / STRONG_BEAR
  - tf_alignment_count  : berapa TF yang align ke satu arah (0–6)
  - ema_fib_score       : khusus EMA Fibonacci alignment semua TF (0–18)

Interpretasi:
  confluence_score > 0.7  → Conviction Mode — sizing besar
  confluence_score 0.3–0.7 → Aligned Mode
  confluence_score < 0.3  → Opportunistic Mode
  tf_alignment_count = 6  → Semua TF sepakat — sinyal paling kuat
```

#### Fibonacci Multi-TF
```
Kalkulasi dari 3 swing berbeda:
  - Macro swing  : 52-week high/low (Daily)
  - Mid swing    : 30-day high/low (4H)
  - Micro swing  : 7-day high/low (1H)

Output  :
  - fib_confluence_zones[] : area di mana 2+ level fib dari swing berbeda overlap
  - strongest_fib_support  : level fib terkuat di bawah harga saat ini
  - strongest_fib_resistance: level fib terkuat di atas harga saat ini
  - fib_cluster_score      : 0–3 (berapa banyak swing yang punya fib di area yang sama)

Interpretasi:
  Fib cluster score 3 = tiga swing punya level di harga yang sama = support/resistance sangat kuat
```

#### VWAP (Volume Weighted Average Price)
```
Periode : Daily VWAP, Weekly VWAP
Output  :
  - vwap_daily         : VWAP sejak open hari ini
  - vwap_weekly        : VWAP sejak open minggu ini
  - price_vs_vwap_daily: ABOVE / BELOW
  - vwap_distance_pct  : % jarak harga dari VWAP daily

Interpretasi:
  Harga di atas VWAP daily = intraday bias bullish
  Harga bounce dari VWAP = support/resistance dinamis kuat
  Scalp long terbaik: harga pullback ke VWAP lalu bounce
```


#### Support & Resistance Detection
```
Metode : Pivot Points + Price Action Levels
Output  :
  - pivot_pp       : Pivot Point harian
  - pivot_r1, r2   : Resistance levels
  - pivot_s1, s2   : Support levels
  - key_levels[]   : Array level penting berdasarkan historical price reaction
  - nearest_support  : Level support terdekat di bawah harga
  - nearest_resistance: Level resistance terdekat di atas harga
  - distance_to_support_pct
  - distance_to_resistance_pct
```

#### Volatility Regime
```
Kalkulasi dari ATR multi-TF + BB Width
Output  :
  - volatility_regime : HIGH / NORMAL / LOW
  - volatility_score  : 0.0–1.0
  - bb_width_percentile: posisi BB width saat ini vs 20-hari historis (%)

Interpretasi:
  Volatility LOW + BB squeeze = breakout imminent, siap-siap
  Volatility HIGH = pasar lagi kencang, perlebar SL
```

### Opsi A Output ke Qwen

```json
{
  "agent": "TeknikalAgent",
  "bias": "LONG | SHORT | NEUTRAL",
  "confidence": 0.0–1.0,
  "confluence_score": -1.0 hingga 1.0,
  "fib_cluster_score": 0–3,
  "volatility_regime": "HIGH | NORMAL | LOW",
  "key_levels": { "support": 69200, "resistance": 71500 },
  "key_points": ["max 4 poin"],
  "risk_flags": ["max 2 flag"]
}
```

---

## 7. OrderflowAgent

### Identitas

| Parameter | Nilai |
|-----------|-------|
| Nama | `OrderflowAgent` |
| Lapor ke | Qwen (Technical Manager) |
| Timeframe | Real-time |
| Tujuan | Analisis kondisi market microstructure: funding, OI, orderbook |
| Loop | Slow Loop + Trigger Loop (jika funding extreme) |

### Tugas Utama

OrderflowAgent membaca "kondisi di balik layar" — siapa yang short, siapa yang long, seberapa besar posisi terbuka, dan apakah ada potensi squeeze. Data ini tidak bisa dilihat dari chart biasa tapi sangat berpengaruh di futures.

### Indikator yang Digunakan

#### Funding Rate
```
Source  : Binance Funding Rate API
Output  :
  - funding_rate_current : % per 8 jam (positif = long bayar short)
  - funding_rate_ma8     : MA 8 periode funding
  - funding_extreme      : bool (|funding| > 0.05%)
  - funding_bias         : LONG_CROWDED (>+0.03%) / SHORT_CROWDED (<-0.03%) / NEUTRAL

Interpretasi:
  funding > +0.05% = terlalu banyak long, potensi long squeeze
  funding < -0.05% = terlalu banyak short, potensi short squeeze
  funding extreme NEGATIF + harga turun = potential bottom (short squeeze incoming)
```

#### Open Interest (OI)
```
Source  : Binance OI API
Output  :
  - oi_current         : OI absolut (dalam BTC)
  - oi_change_4h_pct   : % perubahan OI dalam 4 jam
  - oi_change_24h_pct  : % perubahan OI dalam 24 jam
  - oi_trend           : RISING / FALLING / FLAT
  - price_oi_divergence: bool

Interpretasi:
  Harga naik + OI naik = trend kuat, uang baru masuk
  Harga naik + OI turun = short covering, kurang reliable
  Harga turun + OI naik = trend turun kuat (fresh shorts)
  Harga turun + OI turun = long liquidation, potensi bottom
```

#### Long/Short Ratio
```
Source  : Binance Long/Short Ratio API
Output  :
  - ls_ratio_global    : ratio global long vs short
  - ls_ratio_top_traders: ratio top traders
  - retail_sentiment   : LONG_HEAVY / SHORT_HEAVY / BALANCED

Interpretasi:
  Retail sangat long (>70%) = contrarian bearish signal
  Retail sangat short (>70%) = contrarian bullish signal
  Top traders short + retail long = ikuti top traders
```

#### Orderbook Depth
```
Source  : Binance Orderbook WebSocket (top 20 levels)
Output  :
  - bid_ask_ratio      : total bid volume / total ask volume
  - whale_bid_wall     : level harga dengan bid > 5x average bid
  - whale_ask_wall     : level harga dengan ask > 5x average ask
  - orderbook_imbalance: BUYERS / SELLERS / BALANCED
  - nearest_bid_wall   : harga whale bid terdekat di bawah
  - nearest_ask_wall   : harga whale ask terdekat di atas

Interpretasi:
  Whale bid wall di $69K = support kuat, bagus untuk long
  Whale ask wall di $71K = resistance kuat, TP di bawah itu
```

#### Liquidation Data
```
Source  : Binance Liquidation Stream (WebSocket)
Output  :
  - liq_long_1h_usd    : total long yang dilikuidasi dalam 1 jam
  - liq_short_1h_usd   : total short yang dilikuidasi dalam 1 jam
  - liq_heatmap_above  : area harga di atas dengan banyak liquidation target
  - liq_heatmap_below  : area harga di bawah dengan banyak liquidation target

Interpretasi:
  Banyak short liquidation target di atas = potential squeeze up
  Banyak long liquidation target di bawah = hati-hati, market bisa engineered turun dulu
```

### Opsi A Output ke Qwen

```json
{
  "agent": "OrderflowAgent",
  "bias": "LONG | SHORT | NEUTRAL",
  "confidence": 0.0–1.0,
  "funding_state": "LONG_CROWDED | SHORT_CROWDED | NEUTRAL",
  "squeeze_potential": "LONG_SQUEEZE | SHORT_SQUEEZE | NONE",
  "orderbook_bias": "BUYERS | SELLERS | BALANCED",
  "key_points": ["max 4 poin"],
  "risk_flags": ["max 2 flag"]
}
```

---

## 8. Qwen — Technical Manager

### Identitas

| Parameter | Nilai |
|-----------|-------|
| Model | `qwen2.5:7b` (Ollama lokal) |
| Role | Technical Manager |
| Menerima dari | DataEngine + 5 agent bawahan |
| Mengirim ke | DeepSeek R1 |
| Loop | Slow Loop |

### Tugas Utama

Qwen bukan hanya penerus laporan — dia **menginterpretasi dan mensintesis** kelima laporan agent menjadi satu `technical_report` yang koheren. Dia yang memutuskan apakah sinyal dari berbagai agent saling mendukung atau berkontradiksi.

### Proses

```
1. Terima DataEngine raw data
2. Distribusikan ke 5 agent bawahan (paralel, async)
3. Tunggu semua agent selesai
4. Baca kelima Opsi A report
5. Identifikasi: ada agreement atau konflik?
6. Kompilasi → satu technical_report untuk DeepSeek
```

### Prompt Template

```
SYSTEM:
Kamu adalah Qwen, Technical Manager AI untuk trading BTC/USDT Futures.
Kamu menerima laporan dari 5 agent teknikal bawahan dan mengkompilasi
menjadi satu technical_report yang koheren untuk dikirim ke Chief Supervisor.
Identifikasi agreement dan konflik antar agent. Output JSON valid saja.

USER:
Kompilasi laporan berikut menjadi technical_report:

MacroAgent Report: {macro_report_opsi_a}
TrendAgent Report: {trend_report_opsi_a}
ScalpAgent Report: {scalp_report_opsi_a}
TeknikalAgent Report: {teknikal_report_opsi_a}
OrderflowAgent Report: {orderflow_report_opsi_a}

Trade History Context: {recent_5_trades_summary}

Output JSON:
{
  "overall_bias": "LONG | SHORT | NEUTRAL",
  "overall_confidence": 0.0–1.0,
  "agent_agreement": "FULL | MAJORITY | SPLIT | CONFLICT",
  "market_mode_suggestion": "OPPORTUNISTIC | ALIGNED | CONVICTION",
  "bottom_zone_active": false,
  "key_narrative": "1-2 kalimat ringkas kondisi pasar",
  "agent_summaries": {
    "macro": "...", "trend": "...", "scalp": "...",
    "teknikal": "...", "orderflow": "..."
  },
  "conflicts": ["deskripsi konflik jika ada"],
  "top_risk": "risiko terbesar saat ini"
}
```

---

## 9. Mistral — Sentiment Manager

### Identitas

| Parameter | Nilai |
|-----------|-------|
| Model | `mistral:7b` (Ollama lokal) |
| Role | Sentiment Manager |
| Menerima dari | SentimenEngine + SentimenAgent |
| Mengirim ke | DeepSeek R1 |
| Loop | Slow Loop |

### Prompt Template

```
SYSTEM:
Kamu adalah Mistral, Sentiment Manager AI untuk trading BTC/USDT Futures.
Analisis data sentimen dan berikan laporan ringkas untuk Chief Supervisor.
Output JSON valid saja.

USER:
Analisis sentimen berikut:

Fear & Greed Index: {fng_value} ({fng_label})
Berita terbaru: {news_headlines_5}
Reddit sentiment: {reddit_summary}
Funding Rate: {funding_rate}%
Long/Short Ratio: {ls_ratio}
Trade History Context: {recent_5_trades_summary}

Output JSON:
{
  "sentiment_bias": "BULLISH | BEARISH | NEUTRAL",
  "confidence": 0.0–1.0,
  "fear_greed_interpretation": "...",
  "news_impact": "POSITIVE | NEGATIVE | NEUTRAL",
  "key_narrative": "1-2 kalimat ringkas",
  "risk_events": ["event berita yang perlu diwaspadai"],
  "contrarian_signal": false
}
```

---

## 10. SentimenAgent

### Identitas

| Parameter | Nilai |
|-----------|-------|
| Nama | `SentimenAgent` |
| Lapor ke | Mistral (Sentiment Manager) |
| Sumber Data | Alternative.me, CoinDesk RSS, CoinTelegraph RSS, Reddit RSS, YouTube RSS |
| Tujuan | Kumpulkan dan pre-process data sentimen mentah |
| Loop | Slow Loop |

### Data yang Dikumpulkan

| Sumber | Data | Update |
|--------|------|--------|
| alternative.me/fng | Fear & Greed Index (0–100) | Harian |
| CoinDesk RSS | 5 headline terbaru | Per slow loop |
| CoinTelegraph RSS | 5 headline terbaru | Per slow loop |
| Reddit r/Bitcoin | Top 5 post + sentiment score | Per slow loop |
| Binance API | Funding rate, L/S ratio | Real-time |

Output ke Mistral adalah raw data terstruktur (bukan analisis) — analisis dilakukan oleh Mistral.

---

## 11. DeepSeek R1 — Chief Supervisor

### Identitas

| Parameter | Nilai |
|-----------|-------|
| Model | `deepseek-r1:7b` (Ollama lokal) |
| Role | Chief Supervisor |
| Menerima dari | Qwen (technical_report) + Mistral (sentiment_report) |
| Mengirim ke | ML Engine |
| Loop | Slow Loop |

### Prompt Template

```
SYSTEM:
Kamu adalah DeepSeek R1, Chief Supervisor AI untuk trading BTC/USDT Futures.
Gunakan chain-of-thought reasoning. Pertimbangkan semua laporan dengan cermat.
Output JSON valid saja.

USER:
Buat final_decision berdasarkan:

Technical Report (dari Qwen): {technical_report}
Sentiment Report (dari Mistral): {sentiment_report}
Posisi Aktif: {active_position}
Trade History Summary: {trade_history_batch}

Output JSON:
{
  "market_mode": "OPPORTUNISTIC | ALIGNED | CONVICTION",
  "overall_bias": "LONG | SHORT | NEUTRAL",
  "confidence": 0.0–1.0,
  "bottom_confirmed": false,
  "allow_long": true,
  "allow_short": true,
  "allow_counter_trend": true,
  "sizing_recommendation": "SMALL | MEDIUM | LARGE",
  "tp_modifier": 1.0,
  "sl_modifier": 1.0,
  "reasoning_summary": "ringkasan singkat reasoning",
  "key_risk": "risiko terbesar"
}
```

---

## 12. ML Engine

### Identitas

| Parameter | Nilai |
|-----------|-------|
| Model | XGBoost Ensemble (3 model: Conservative, Balanced, Aggressive) |
| Menerima dari | DeepSeek R1 (final_decision) + DataEngine (OHLCV real-time) |
| Mengirim ke | Executor + Claude API |
| Loop | Fast Loop |

### Tugas Utama

ML Engine menerima `final_decision` dari DeepSeek sebagai **konteks dan filter**, lalu secara independen mencari area entry terbaik menggunakan fitur teknikal dari semua timeframe.

### Input Features (Opsi B — Detail Penuh)

ML Engine menggunakan **semua nilai numerik** dari kelima agent. Feature set lengkap didefinisikan di `MODELS.md`.

### Output — Entry Proposal

```json
{
  "direction": "LONG | SHORT | NONE",
  "entry_price": 69850.0,
  "sl_price": 69757.0,
  "tp_price": 70010.0,
  "sl_pct": 0.0013,
  "tp_pct": 0.0023,
  "sizing_mode": "SMALL | MEDIUM | LARGE",
  "confidence_conservative": 0.61,
  "confidence_balanced": 0.68,
  "confidence_aggressive": 0.74,
  "ensemble_confidence": 0.68,
  "detail_data": { ... }
}
```

---

## 13. Claude API — Entry Assistant

### Identitas

| Parameter | Nilai |
|-----------|-------|
| Model | Claude API (cloud) |
| Role | Entry Assistant — opsional |
| Menerima dari | ML Engine (entry_proposal + detail_data) |
| Mengirim ke | Executor |
| Loop | Fast Loop (jika aktif) |
| Fallback | Executor pakai entry_proposal ML langsung + cached DeepSeek decision |

### Tugas Utama

Claude **tidak menentukan arah pasar** — itu sudah dilakukan DeepSeek. Claude hanya memeriksa apakah area entry yang ditemukan ML **masuk akal secara kontekstual**: apakah TP realistis, apakah SL tidak terlalu ketat, apakah ada alasan untuk adjust.

### Prompt Template

```
SYSTEM:
Kamu adalah Entry Assistant AI untuk trading BTC/USDT Futures.
Kamu menerima entry proposal dari ML Engine dan memvalidasinya.
Kamu TIDAK menentukan arah pasar — itu sudah ditentukan oleh Chief Supervisor.
Tugasmu: apakah entry ini masuk akal? Perlu adjust TP/SL?
Output JSON valid saja. Jika entry tidak layak, berikan alasan singkat.

USER:
Validasi entry proposal berikut:

Entry Proposal: {entry_proposal}
Market Context (dari DeepSeek): {final_decision_summary}
Sentiment Summary: {sentiment_report_summary}
Recent Trade History: {last_3_trades}
Kondisi Teknikal Saat Ini: {scalp_report_opsi_b}

Output JSON:
{
  "approved": true,
  "adjusted_entry": 69850.0,
  "adjusted_tp": 70060.0,
  "adjusted_sl": 69757.0,
  "confidence_boost": 0.05,
  "note": "TP diperlebar sedikit karena momentum kuat",
  "reject_reason": null
}
```

### Fallback jika Claude Tidak Aktif

```python
if claude_active and claude_response.approved:
    final_entry = claude_response.adjusted_entry
    final_tp    = claude_response.adjusted_tp
    final_sl    = claude_response.adjusted_sl
else:
    # Gunakan ML proposal langsung
    final_entry = ml_proposal.entry_price
    final_tp    = ml_proposal.tp_price
    final_sl    = ml_proposal.sl_price
```

---

## 14. Data Flow Lengkap

```
SLOW LOOP (setiap 5–10 menit):

DataEngine ──────────────────────────── SentimenEngine
     │                                        │
     ▼                                        ▼
[Load Qwen]                             [Load Mistral]
  Distribusikan ke 5 agent               SentimenAgent kumpul data
  Tunggu semua agent selesai             Mistral analisis
  Kompilasi → technical_report           → sentiment_report
[Unload Qwen]                           [Unload Mistral]
     │                                        │
     └──────────────┬─────────────────────────┘
                    ▼
             [Load DeepSeek R1]
             Terima kedua report
             Chain-of-thought reasoning
             → final_decision (cached TTL 15 menit)
             [Unload DeepSeek R1]
                    │
                    ▼
             ML Engine
             Cari entry_proposal
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   [Claude API]          Executor Standby
   Validasi entry        (pakai cache)
   → adjusted entry
          │
          ▼
   Executor Eksekusi

FAST LOOP (setiap ~5 detik):
  Monitor posisi aktif
  SL/TP check
  Trailing stop update
  Auto-flip check
  → Gunakan cached final_decision

TRIGGER LOOP (event-driven):
  Dipicu saat: price spike >1%, funding extreme, breaking news
  → Paksa slow loop cycle baru dengan prioritas tinggi

FEEDBACK LOOP (setiap N trade):
  Post-Trade Analyzer kompilasi hasil
  → Kirim batch summary ke Qwen, DeepSeek, Mistral, Claude
```

---
