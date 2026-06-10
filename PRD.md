# Product Requirements Document
# AI Crypto Futures Trading Bot — v4.0
### Opportunistic Hierarchical Swarm System

> **Status:** Active Development  
> **Versi:** v4.2 *(revised architecture — Qwen as Technical Manager, Claude as Entry Assistant)*  
> **Tanggal:** 2026-06-03  
> **Symbol:** BTC/USDT:USDT — Binance USDT-M Futures  
> **Leverage:** 100x (Testnet / Small Capital)  
> **Hardware:** Intel Core 5 215H | RTX 5050 8GB VRAM | 16GB RAM  

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Tujuan & Ruang Lingkup](#2-tujuan--ruang-lingkup)
3. [Arsitektur Sistem](#3-arsitektur-sistem)
4. [Logic Inti](#4-logic-inti)
5. [LLM Stack & Manajemen VRAM](#5-llm-stack--manajemen-vram)
6. [Risk Management](#6-risk-management)
7. [Data Sources & Sentimen](#7-data-sources--sentimen)
8. [Struktur Kode & Direktori](#8-struktur-kode--direktori)
9. [Perbandingan v3.1 vs v4.0](#9-perbandingan-v31-vs-v40)
10. [Rencana Pengembangan](#10-rencana-pengembangan)
11. [Risiko & Mitigasi](#11-risiko--mitigasi)
12. [Konfigurasi Parameter](#12-konfigurasi-parameter)
13. [Cara Menjalankan](#13-cara-menjalankan)
14. [Glosarium](#14-glosarium)

---

## 1. Ringkasan Eksekutif

Proyek ini adalah generasi keempat (v4.0) dari platform trading otomatis berbasis AI untuk instrumen BTC/USDT Futures di Binance. Versi ini memperkenalkan paradigma baru: **Opportunistic Hierarchical Swarm** — sistem di mana setiap agent memiliki spesialisasi timeframe dan domain analisis yang jelas, namun tetap mampu beroperasi secara agresif dan fleksibel dalam kondisi pasar apapun.

**Filosofi utama:** Bot tidak boleh menganggur. Saat makro belum memberikan sinyal tegas, bot memanfaatkan peluang scalping jangka pendek di kedua arah (long/short). Saat konfirmasi makro tiba, bot mengalign posisi, memperbesar sizing, dan memperlebar target profit.

**Fokus v4.0:** Core trading engine yang presisi — eksekusi, manajemen posisi, dan keputusan AI yang akurat. Dashboard monitoring dikerjakan di fase berikutnya setelah bot dapat trading dengan andal.

### Upgrade Utama dari v3.1

| # | Upgrade | Dampak |
|---|---------|--------|
| 1 | Arsitektur multi-timeframe hierarkis (Weekly/Daily → 4H/1H → 15M/5M) | Konteks pasar jauh lebih kaya |
| 2 | LLM stack baru: Qwen sebagai Technical Manager + agent hierarki | Analisis multi-TF terstruktur dan terpusat |
| 3 | Sequential LLM loading (bukan paralel) | Efisien di 8GB VRAM |
| 4 | DeepSeek R1 sebagai Chief Supervisor | Final decision berdasarkan Qwen + Mistral report |
| 5 | Claude API sebagai Entry Assistant (opsional) | Validasi area entry ML — bisa dimatikan kapan saja |
| 6 | Dynamic TP/SL berbasis mode operasi | TP diperlebar saat macro kuat |
| 7 | Fast Loop + Slow Loop separation | LLM delay tidak blokir eksekusi |

---

## 2. Tujuan & Ruang Lingkup

### 2.1 Tujuan Proyek

| No | Tujuan | Metrik Keberhasilan |
|----|--------|---------------------|
| 1 | Bot aktif mencari peluang di setiap kondisi pasar | Tidak ada idle time selama market jam aktif |
| 2 | Integrasi analisis multi-timeframe hierarkis | Entry dikonfirmasi minimal dari 2 timeframe |
| 3 | 3 LLM spesialis untuk keputusan kontekstual | Supervisor menghasilkan JSON valid > 95% |
| 4 | LLM efisien di hardware konsumer | Peak VRAM < 6GB per model |
| 5 | Meningkatkan win rate vs v3.1 | Target win rate > 55% setelah 200 trade testnet |

### 2.2 Ruang Lingkup v4.0

- Instrumen: BTC/USDT:USDT Binance USDT-M Futures
- Leverage: 100x (testnet / modal kecil)
- Timeframe analisis: Weekly, Daily, 4H, 1H, 15M, 5M
- LLM: Qwen 2.5:7b, Mistral:7b, DeepSeek R1:7b (lokal via Ollama)
- Data sentimen: gratis (RSS, Fear & Greed API, Reddit, YouTube RSS)
- Output utama: eksekusi order yang presisi dan manajemen posisi yang andal

### 2.3 Di Luar Ruang Lingkup v4.0

- Dashboard / visualisasi UI *(fase berikutnya)*
- Multi-asset / multi-exchange *(fase berikutnya)*
- Integrasi API LLM berbayar (Grok, GPT-4, dll.)

---

## 3. Arsitektur Sistem

### 3.1 Gambaran Umum

```
INPUT
  DataEngine (OHLCV · OB · Funding · OI)
  SentimenEngine (RSS · Fear&Greed · Reddit)
        │                        │
        ▼                        ▼
┌───────────────────┐    ┌───────────────────┐
│  LAYER 1A         │    │  LAYER 1B         │
│  Qwen 2.5:7b      │    │  Mistral:7b       │
│  Technical Mgr    │    │  Sentiment Mgr    │
│  ┌─────────────┐  │    │  ┌─────────────┐  │
│  │ MacroAgent  │  │    │  │SentimenAgent│  │
│  │ TrendAgent  │◄─┤    │  └─────────────┘  │
│  │ ScalpAgent  │  │    └────────┬──────────┘
│  │TeknikalAgent│  │             │
│  └──────┬──────┘  │             │
│  report ▲▼ ke     │             │
│  Qwen kompilasi   │             │
└─────────┬─────────┘             │
          │ technical_report       │ sentiment_report
          ▼                        ▼
┌──────────────────────────────────────────┐
│  LAYER 2: DeepSeek R1:7b — Chief Supervisor │
│  Chain-of-thought · Weigh semua laporan  │
│  "Bottom confirmed? Opportunistic/Conviction?" │
│  → final_decision (cached, TTL 15 menit) │
└─────────────────┬────────────────────────┘
                  │ final_decision
                  ▼
┌──────────────────────────────────────────┐
│  LAYER 3: ML Engine — XGBoost Ensemble   │
│  Terima final_decision dari DeepSeek     │
│  Cari area entry konkret (harga, zone)   │
│  → entry_proposal (harga · SL · TP)      │
└──────────┬───────────────────────────────┘
           │ entry_proposal
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌──────────────────────────────┐
│Executor│  │ Claude API — Entry Assistant  │
│Standby │◄─│ Validasi entry · TP · SL     │
│        │  │ (opsional, bisa dimatikan)    │
└───┬────┘  └──────────────────────────────┘
    │ jika Claude off → pakai cached DeepSeek decision
    ▼
┌──────────────────────────────────────────┐
│  LAYER 4: Risk Guardian                  │
│  Kelly · Fee guardrail · Circuit breaker │
└─────────────────┬────────────────────────┘
                  ▼
┌──────────────────────────────────────────┐
│  EKSEKUSI                                │
│  Position Manager → Order Router         │
│  → Binance USDT-M Futures                │
└─────────────────┬────────────────────────┘
                  │ hasil trade
                  ▼
┌──────────────────────────────────────────┐
│  FEEDBACK LOOP                           │
│  Simpan History → Post-Trade Analyzer    │
│  → dikirim ke Qwen, DeepSeek, Mistral,   │
│    Claude (batch, setiap N trade)        │
└──────────────────────────────────────────┘
```

### 3.2 Layer 1A — Qwen sebagai Technical Manager

Qwen bukan hanya LLM analisis biasa — dia adalah **manajer hierarki teknikal**. DataEngine mengirim semua data ke Qwen, lalu Qwen mendistribusikan ke 4 agent bawahan untuk dianalisa per domain. Setelah semua agent selesai, hasil analisa naik kembali ke Qwen untuk dikompilasi menjadi satu `technical_report` yang dikirim ke DeepSeek.

```
DataEngine → Qwen
                ├── distribusikan ke → MacroAgent   → hasil naik ke Qwen
                ├── distribusikan ke → TrendAgent   → hasil naik ke Qwen
                ├── distribusikan ke → ScalpAgent   → hasil naik ke Qwen
                └── distribusikan ke → TeknikalAgent → hasil naik ke Qwen
                          ↓
                Qwen kompilasi semua hasil
                "Weekly/Daily = bottom zone $60K"
                "15M momentum long terdeteksi"
                          ↓
                technical_report → DeepSeek
```

| Agent | Timeframe | Tugas Utama | Output ke Qwen |
|-------|-----------|-------------|----------------|
| `MacroAgent` | Weekly / Daily | Regime pasar, S/R mayor, estimasi bottom/top | `macro_summary` |
| `TrendAgent` | 4H / 1H | EMA alignment, MACD momentum, premium/discount zone | `trend_summary` |
| `ScalpAgent` | 15M / 5M | RSI divergence, Bollinger Bands, volume spike, order block | `scalp_summary` |
| `TeknikalAgent` | Multi-TF | 50+ indikator teknikal, confluence score | `technical_summary` |

### 3.3 Layer 1B — Mistral sebagai Sentiment Manager

SentimenEngine mengirim data ke Mistral secara independen dari jalur Qwen. Mistral mengelola SentimenAgent dan menghasilkan `sentiment_report` yang dikirim langsung ke DeepSeek.

| Komponen | Tugas | Output |
|----------|-------|--------|
| `SentimenAgent` | Fear & Greed, RSS berita, Reddit, funding rate | `sentiment_raw` |
| `Mistral:7b` | Kompilasi dan interpretasi sentimen | `sentiment_report` |

### 3.4 Layer 2 — DeepSeek R1 sebagai Chief Supervisor

DeepSeek menerima `technical_report` dari Qwen dan `sentiment_report` dari Mistral, lalu menghasilkan `final_decision` menggunakan chain-of-thought reasoning.

| Input | Sumber |
|-------|--------|
| `technical_report` | Qwen (kompilasi 4 agent) |
| `sentiment_report` | Mistral (SentimenAgent) |
| `trade_history_batch` | Post-Trade Analyzer (feedback loop) |

Output `final_decision`:

```json
{
  "macro_bias": "LONG | SHORT | NEUTRAL",
  "market_mode": "OPPORTUNISTIC | ALIGNED | CONVICTION",
  "bottom_confirmed": true,
  "confidence": 0.78,
  "reasoning_summary": "Weekly menunjukkan area bottom $60K...",
  "suggested_direction": "LONG",
  "allow_counter_trend": true
}
```

> `final_decision` di-cache dengan TTL 15 menit. Fast Loop menggunakan cache ini tanpa memanggil LLM kembali.

### 3.5 Layer 3 — ML Engine

ML Engine menerima `final_decision` dari DeepSeek sebagai **konteks**, lalu mencari area entry konkret menggunakan XGBoost ensemble.

```
final_decision (dari DeepSeek)
        +
OHLCV real-time (dari DataEngine)
        ↓
XGBoost Conservative + Balanced + Aggressive
        ↓
entry_proposal:
  - entry_price: $69,850
  - sl_price:    $69,757  (0.13%)
  - tp_price:    $70,010  (0.23%)
  - sizing_mode: MEDIUM
  - direction:   LONG
```

### 3.6 Claude API — Entry Assistant (Opsional)

Claude menerima `entry_proposal` dari ML Engine dan `sentiment_report` dari Mistral, lalu **memvalidasi apakah entry tersebut layak** dari sisi kontekstual.

**Claude adalah entry assistant, bukan supervisor.** Dia tidak menentukan arah pasar — itu sudah dilakukan DeepSeek. Tugasnya hanya memastikan area entry yang ditemukan ML masuk akal secara kontekstual.

```
entry_proposal + sentiment_report + final_decision (context)
        ↓
Claude API
"Entry $69,850 masuk akal? TP $70,010 cukup atau bisa lebih jauh?"
        ↓
entry_confirmation:
  - approved: true
  - adjusted_tp: $70,050  (sedikit diperlebar)
  - adjusted_sl: $69,757  (tetap)
  - note: "momentum kuat, TP bisa diperlebar sedikit"
```

**Jika Claude dimatikan atau API gagal:**
- Executor menggunakan `entry_proposal` dari ML Engine langsung
- Divalidasi hanya oleh `final_decision` cached dari DeepSeek
- Sistem tetap berjalan normal tanpa gangguan

### 3.7 Layer 4 — Eksekusi & Risk

| Kondisi Pasar | Mode | Sizing | TP Target | SL |
|---------------|------|--------|-----------|-----|
| Macro UNCLEAR / NEUTRAL | **Opportunistic** | Medium (3–5%) | 0.20% | 0.12% |
| Macro LONG terkonfirmasi | **Aligned Long** | Medium-Large (5–8%) | 0.40% | 0.15% |
| Macro SHORT terkonfirmasi | **Aligned Short** | Medium-Large (5–8%) | 0.40% | 0.15% |
| Confluence sangat tinggi | **Conviction** | Besar (8–15%) | ATR-based | 0.15% + trailing |
| Counter-trend (melawan macro) | **Counter-Trend** | Kecil fixed (1–3%) | 0.15% | 0.10% |

---

## 4. Logic Inti

### 4.1 Opportunistic Logic — Bot Tidak Pernah Menganggur

```python
# Pseudocode Chief Manager Decision

if macro_bias == "NEUTRAL" or macro_bias == "UNCLEAR":
    mode = OPPORTUNISTIC
    allow_long = True
    allow_short = True
    sizing = MEDIUM        # 3–5% — cukup besar agar fee tidak makan profit
    tp = TP_TIGHT          # 0.20%

elif macro_bias == "LONG":
    mode = ALIGNED_LONG
    allow_long = True      # medium-large sizing (5–8%)
    allow_short = True     # counter-trend: small fixed (1–3%), TP ketat 0.15%
    on_short_tp: auto_flip_to_long()

elif macro_bias == "SHORT":
    mode = ALIGNED_SHORT
    allow_short = True     # medium-large sizing (5–8%)
    allow_long = True      # counter-trend: small fixed (1–3%), TP ketat 0.15%
    on_long_tp: auto_flip_to_short()

if confluence_score > CONVICTION_THRESHOLD:
    mode = CONVICTION
    sizing = LARGE         # 8–15% — ini moment yang ditunggu
    tp = ATR_DYNAMIC
    activate_trailing_stop()
```

### 4.2 Auto-Flip Logic

Bot tidak melewatkan momentum setelah menutup posisi counter-trend:

| Scenario | Aksi Saat Close | Kondisi Auto-Flip | Aksi Flip |
|----------|-----------------|-------------------|-----------|
| Short TP tercapai, macro = LONG | Close SHORT | macro_bias == LONG AND scalp_signal valid | Immediately open LONG |
| Long TP tercapai, macro = SHORT | Close LONG | macro_bias == SHORT AND scalp_signal valid | Immediately open SHORT |
| Posisi kena SL | Close posisi | Tidak ada auto-flip setelah SL | Tunggu sinyal baru |
| AI Confidence Loss | Close posisi | Evaluasi ulang semua layer | Tunggu konfirmasi baru |

### 4.3 Macro Confirmation Rules

`macro_bias` dianggap **TERKONFIRMASI** jika **minimal 3 dari 6 kondisi** terpenuhi, dengan **minimal 1 kondisi berbobot TINGGI**:

| Kondisi | Definisi Programatik | Bobot |
|---------|---------------------|-------|
| Price Bounce | Harga bounce ≥ 2x dari level yang sama dalam 24 jam | TINGGI |
| Volume Spike | Volume candle > 2x rata-rata 20 candle saat menyentuh level | TINGGI |
| RSI Divergence | Bullish/Bearish divergence di timeframe 4H | MEDIUM |
| Funding Rate Extreme | Funding rate > +0.05% atau < -0.05% | MEDIUM |
| EMA Alignment | EMA 20 > 50 > 200 (bullish) atau sebaliknya di 4H | MEDIUM |
| Orderbook Whale Wall | Bid/Ask wall > 5x rata-rata depth di level kunci | MEDIUM |

### 4.4 Fast Loop vs Slow Loop

Pemisahan dua loop ini krusial agar LLM tidak memblokir eksekusi order:

| Loop | Frekuensi | Tugas | LLM? |
|------|-----------|-------|------|
| **Fast Loop** | Setiap candle (~5 detik) | Eksekusi order, monitor posisi, trailing stop, SL/TP check | ❌ Pakai cache |
| **Slow Loop** | Setiap 5–10 menit | Jalankan LLM cycle, update composite score, refresh macro bias | ✅ Sequential |
| **Trigger Loop** | Event-driven | Dipicu saat: price spike >1%, volume anomaly, funding extreme, breaking news | ✅ Prioritas tinggi |

```
Timeline contoh:

t=0:00  Slow Loop mulai → load Qwen
t=1:15  Qwen selesai → unload → load Mistral
t=2:30  Mistral selesai → unload → load DeepSeek R1
t=4:00  DeepSeek selesai → final_decision di-cache → unload
t=4:05  Fast Loop update → pakai keputusan baru

Selama t=0:00 hingga t=4:00, Fast Loop tetap aktif
menggunakan keputusan dari siklus sebelumnya.
```

### 4.5 AI Confidence Loss

Mekanisme dari v3.1 yang dipertahankan: jika evaluasi AI berubah menjadi `HOLD` setelah posisi terbuka lebih dari 5 menit, posisi ditutup untuk meminimalisir risiko paparan yang tidak termonitor.

---

## 5. LLM Stack & Manajemen VRAM

### 5.1 Spesifikasi Model

| Model | Ukuran File | Est. VRAM | Role | Keunggulan |
|-------|------------|-----------|------|------------|
| `qwen2.5:7b` | ~4.7 GB | ~4.5 GB | **Technical Manager** | Mendistribusikan ke 4 agent, mengkompilasi hasil, mathematical reasoning kuat |
| `mistral:7b` | ~4.1 GB | ~4.5 GB | **Sentiment Manager** | Mengelola SentimenAgent, contextual understanding kuat |
| `deepseek-r1:7b` | ~5.0 GB | ~5.0 GB | **Chief Supervisor** | Chain-of-thought reasoning, final decision maker |
| `Claude API` | Cloud | — | **Entry Assistant** | Validasi entry_proposal dari ML, opsional, bisa dimatikan |

> **Mengapa tiga model berbeda keluarga?** Qwen (Alibaba), Mistral (Mistral AI), DeepSeek (DeepSeek AI) — masing-masing dilatih dengan arsitektur dan data berbeda. Ini memastikan perspektif yang genuinely berbeda, bukan echo chamber seperti jika menggunakan dua model dari keluarga Llama.

### 5.2 Sequential Loading Protocol

```
Trigger LLM Cycle (Slow Loop)
       │
       ▼
┌─────────────────┐
│ Load Qwen 2.5:7b│
│ Distribusikan   │
│ ke 4 agent      │
│ Kompilasi hasil │
│ → technical_    │
│   report        │
│ Unload          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Load Mistral:7b │
│ Kelola          │
│ SentimenAgent   │
│ → sentiment_    │
│   report        │
│ Unload          │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ Load DeepSeek R1:7b  │
│ Terima kedua report  │
│ Chain-of-thought     │
│ → final_decision     │
│ Update cache         │
│ Unload               │
└──────────┬───────────┘
           │ final_decision → ML Engine
           ▼
┌──────────────────────┐
│ ML Engine            │
│ Cari entry_proposal  │
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌────────────────┐
│Executor │  │ Claude API     │
│standby  │◄─│ (jika aktif)   │
└─────────┘  └────────────────┘
```

**Estimasi total per cycle: 3–5 menit**

### 5.3 Prompt Engineering

#### Qwen — Technical Analyst

```
SYSTEM:
Kamu adalah Technical Analyst AI untuk trading BTC/USDT Futures.
Analisis data teknikal yang diberikan dan hasilkan laporan dalam format JSON.
Fokus pada matematika dan indikator. Jangan spekulasi di luar data.
Selalu output JSON valid tanpa preamble atau markdown.

USER:
Analisis data berikut:
- OHLCV 5M (20 candle terakhir): {ohlcv_5m}
- OHLCV 1H (20 candle terakhir): {ohlcv_1h}
- RSI(14): {rsi} | EMA 20/50/200: {ema} | MACD: {macd}
- Bollinger Bands: {bb} | ATR(14): {atr}
- ML Score (Conservative/Balanced/Aggressive): {ml_scores}
- Multi-TF EMA alignment score: {tf_alignment}

Output JSON:
{
  "bias": "LONG|SHORT|NEUTRAL",
  "confidence": 0.0-1.0,
  "key_reasons": ["reason1", "reason2"],
  "risk_factors": ["risk1"],
  "suggested_tp_pct": 0.0025,
  "suggested_sl_pct": 0.0013
}
```

#### Mistral — Sentiment & Macro Analyst

```
SYSTEM:
Kamu adalah Sentiment & Macro Analyst AI untuk trading BTC/USDT.
Interpretasi berita, sentimen pasar, dan data makro, lalu hasilkan JSON.
Selalu output JSON valid tanpa preamble atau markdown.

USER:
Analisis konteks pasar berikut:
- Fear & Greed Index: {fng_value} ({fng_label})
- Berita terbaru (5 terakhir): {news_headlines}
- Funding Rate: {funding_rate}%
- Open Interest trend: {oi_trend} ({oi_change_pct}% dalam 4 jam)
- Long/Short Ratio: {ls_ratio}
- Macro bias saat ini dari MacroAgent: {macro_bias}

Output JSON:
{
  "sentiment_bias": "BULLISH|BEARISH|NEUTRAL",
  "confidence": 0.0-1.0,
  "key_narratives": ["narrative1"],
  "risk_events": ["event1"],
  "macro_alignment": true,
  "funding_interpretation": "string"
}
```

#### DeepSeek R1 — Supervisor

```
SYSTEM:
Kamu adalah Chief Trading Supervisor AI. Terima laporan dari dua analis
dan buat keputusan trading akhir. Gunakan chain-of-thought reasoning.
Pertimbangkan konflik sinyal dengan cermat. Output JSON valid, tanpa preamble.

USER:
Buat keputusan berdasarkan:
- Technical Report: {technical_report}
- Sentiment Report: {sentiment_report}
- Composite Score: {composite_score} | Bias: {composite_bias}
- Mode Pasar Saat Ini: {market_mode}
- Posisi Aktif: {active_position} (None jika tidak ada)
- Macro Confirmed: {macro_confirmed}

Output JSON:
{
  "action": "LONG|SHORT|HOLD|CLOSE",
  "confidence": 0.0-1.0,
  "sizing_mode": "SMALL|MEDIUM|LARGE",
  "tp_modifier": 1.0,
  "sl_modifier": 1.0,
  "auto_flip": false,
  "reasoning_summary": "string singkat"
}
```

### 5.4 Fallback Jika LLM Gagal

Jika parsing JSON gagal atau timeout tercapai:

```python
# Fallback hierarchy:
# 1. Retry inference 1x dengan temperature lebih rendah
# 2. Jika masih gagal → gunakan cached decision dari cycle sebelumnya
# 3. Jika cache expired (> 15 menit) → fallback ke rule-based composite score
# 4. Log error ke ea.log untuk post-mortem
```

---

## 6. Risk Management

### 6.1 Parameter Risk Utama

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `LEVERAGE` | 100 | Testnet / small capital |
| `MAX_DAILY_LOSS_PCT` | 2% | Circuit breaker harian |
| `MAX_DRAWDOWN_PCT` | 10% | Circuit breaker kumulatif |
| `KELLY_WINDOW` | 7 | Jumlah trade terakhir untuk basis Kelly |
| `KELLY_MAX_FRACTION` | 0.50 | Half-Kelly cap |
| `MARGIN_OPPORTUNISTIC` | 3–5% | Cukup besar agar fee tidak makan profit |
| `MARGIN_ALIGNED` | 5–8% | Ada konfirmasi arah, layak lebih besar |
| `MARGIN_CONVICTION` | 8–15% | Moment terbesar, sizing maksimal |
| `MARGIN_COUNTER_TREND` | 1–3% fixed | Selalu kecil tanpa kompromi |
| `TP_OPPORTUNISTIC` | 0.20% | ~20% ROI di 100x |
| `TP_ALIGNED` | 0.40% | ~40% ROI di 100x |
| `TP_COUNTER_TREND` | 0.15% | Ambil cepat, langsung keluar |
| `TP_CONVICTION` | ATR-based | Dinamis, trailing stop aktif |
| `SL_OPPORTUNISTIC` | 0.12% | Hard stop opportunistic |
| `SL_ALIGNED` | 0.15% | Hard stop aligned |
| `SL_COUNTER_TREND` | 0.10% | Paling ketat — melawan arus |
| `TRAILING_ATR_MULT` | 0.50 | Trailing stop = 0.5x ATR 5M |
| `REVERSAL_MIN_AGE_MIN` | 5 menit | Usia minimal posisi sebelum di-flip |
| `REVERSAL_BYPASS_CONF` | 0.85 | Bypass batas usia jika confidence sangat tinggi |
| `FEE_GUARDRAIL_MULT` | 1.50 | Batalkan trade jika fee > 150% SL risk |
| `FEE_RATIO_MAX` | 0.25 | Batalkan trade jika fee > 25% expected profit |

### 6.2 Dynamic Sizing Per Mode

| Mode | Margin % | TP | SL | Logic |
|------|----------|----|----|-------|
| Opportunistic | 3–5% | 0.20% | 0.12% | Medium sizing — cukup besar agar fee tidak makan profit, tapi macro belum terkonfirmasi |
| Aligned | 5–8% | 0.40% | 0.15% | Medium-large — ada konfirmasi arah, risiko lebih terukur |
| Conviction | 8–15% | ATR-based | 0.15% + trailing | Terbesar — multi-TF confluence tinggi, ini trade yang "serius" |
| Counter-Trend | 1–3% fixed | 0.15% | 0.10% | Selalu minimal tanpa kompromi — melawan arus, ambil cepat lalu keluar |

> **Prinsip sizing v4.0:** Sizing terbesar bukan di scalping, tapi di Conviction Mode. Counter-trend selalu paling kecil. Opportunistic dinaikkan dari v3.1 (1–3% → 3–5%) agar efisien secara fee.

### 6.3 Circuit Breaker Hierarchy

| Level | Trigger | Aksi |
|-------|---------|------|
| **Warning** | Daily loss > 1% | Kurangi sizing 50%, catat log |
| **Throttle** | Daily loss > 1.5% | Hanya opportunistic mode, sizing minimum |
| **Stop** | Daily loss > 2% ATAU drawdown > 10% | Tutup semua posisi, berhenti trading hari ini |
| **Emergency** | Exchange error > 3x retry, posisi tidak bisa ditutup | Halt sistem, log seluruh state |

---

## 7. Data Sources & Sentimen

### 7.1 Market Data

| Sumber | Data | Library | Biaya |
|--------|------|---------|-------|
| Binance USDT-M Futures | OHLCV (5M/15M/1H/4H/1D/Weekly) | `ccxt` | Gratis |
| Binance Public API | Orderbook, OI, Funding Rate, L/S Ratio | `ccxt` | Gratis |
| Binance WebSocket | Tick data real-time, liquidation stream | `ccxt.pro` | Gratis |

### 7.2 Sentiment Data (Semua Gratis)

| Sumber | Data | Endpoint |
|--------|------|----------|
| Alternative.me | Fear & Greed Index | `https://api.alternative.me/fng/` |
| CoinDesk RSS | Berita kripto | `https://www.coindesk.com/arc/outboundfeeds/rss/` |
| CoinTelegraph RSS | Berita kripto | `https://cointelegraph.com/rss` |
| Reddit RSS | Sentimen komunitas | `https://www.reddit.com/r/Bitcoin/.rss` |
| YouTube RSS | Konten creator kripto besar | `https://www.youtube.com/feeds/videos.xml?channel_id={ID}` |

### 7.3 Feature Engineering (50+ Fitur)

| Kategori | Indikator |
|----------|-----------|
| **Trend** | EMA (Custom TF); Linear Regression; Swing Structure |
| **Momentum** | RSI 14; MACD (12,26,9) |
| **Volatility** | Bollinger Bands (20,2); ATR 14; Keltner Channel; Historical Volatility |
| **Volume** | Volume MA; OBV; VWAP; Volume Delta; Cumulative Delta |
| **Multi-TF Confluence** | RSI alignment score (5M vs 1H vs 4H); EMA trend alignment score |
| **Orderflow** | Bid/Ask imbalance; Whale wall detection; Liquidation heatmap proxy |
| **Derived** | Price distance from S/R; Candle pattern encoding; Session filter |

---

## 8. Struktur Kode & Direktori

```
ai-trading-v4/
├── main.py                          # Entry point utama
├── config.py                        # Semua parameter terpusat
├── validate.py                      # Validasi environment & koneksi
├── .env                             # API keys (tidak di-commit)
├── requirements.txt
│
├── core/
│   ├── agents/
│   │   ├── macro_agent.py           # Daily/Weekly — regime & S/R mayor
│   │   ├── trend_agent.py           # 4H/1H — momentum & EMA alignment
│   │   ├── scalp_agent.py           # 15M/5M — entry timing & local signal
│   │   ├── sentiment_agent.py       # Berita RSS + Fear & Greed
│   │   └── orderflow_agent.py       # Funding rate + OI + orderbook depth
│   │
│   ├── llm/
│   │   ├── llm_loader.py            # Sequential load/unload via Ollama API
│   │   ├── technical_llm.py         # Qwen 2.5:7b wrapper + prompt
│   │   ├── sentiment_llm.py         # Mistral:7b wrapper + prompt
│   │   ├── supervisor_llm.py        # DeepSeek R1:7b wrapper + prompt
│   │   └── llm_cache.py             # Cache final_decision + TTL
│   │
│   ├── feature_engine.py            # 50+ fitur teknikal dari OHLCV
│   ├── ml_engine.py                 # XGBoost ensemble (3 model)
│   ├── composite_scorer.py          # Weighted aggregation semua sinyal
│   ├── chief_manager.py             # Logic opportunistic + auto-flip + mode
│   ├── execution_engine.py          # Order placement via ccxt (limit/market)
│   ├── position_manager.py          # State machine posisi + trailing stop
│   ├── risk_manager.py              # Kelly + circuit breaker + fee guardrail
│   ├── trade_memory.py              # SQLite trade history (48 jam rolling)
│   └── post_trade_analyzer.py       # LLM feedback loop pasca trade
│
├── loops/
│   ├── fast_loop.py                 # Setiap candle (~5 detik)
│   ├── slow_loop.py                 # Setiap 5–10 menit (LLM cycle)
│   └── trigger_loop.py              # Event-driven (spike, anomaly, news)
│
├── data_pipeline/
│   ├── market_data_engine.py        # Fetch OHLCV multi-TF dari Binance
│   ├── synthetic_augmentation.py    # 50% real + 50% augmented (Gaussian noise)
│   └── sentiment_fetcher.py         # RSS parser + Fear & Greed poller
│
├── models/
│   ├── xgb_conservative.json
│   ├── xgb_balanced.json
│   └── xgb_aggressive.json
│
├── data/
│   ├── synthetic_ohlcv.csv          # Dataset training (100K candles)
│   └── sentiment_cache.json         # Cache sentimen terakhir
│
└── logs/
    ├── trading.db                   # SQLite — semua sinyal & trade
    ├── trade_memory.db              # SQLite — feedback loop LLM
    ├── llm_cache.json               # Cache keputusan LLM terbaru
    └── ea.log                       # Rotating text log
```

### 8.1 Status Komponen vs v3.1

| Komponen | Status | Catatan |
|----------|--------|---------|
| `MacroAgent`, `TrendAgent`, `ScalpAgent` | **BARU** | Menggantikan single-TF signal engine |
| `OrderflowAgent` | **BARU** | Konsolidasi OI + funding + depth |
| `chief_manager.py` | **BARU** | Logic opportunistic + auto-flip |
| `llm_loader.py` (sequential) | **BARU** | Load/unload otomatis per model |
| `DeepSeek R1` supervisor | **BARU** | Gantikan llama3.1 sebagai decision maker |
| `Mistral:7b` sentiment | **BARU** | Gantikan llama3.2 |
| `loops/` (fast + slow + trigger) | **BARU** | Kritis untuk responsivitas |
| `Qwen 2.5:7b` technical | **UPGRADE** | Dari 3b → 7b, akurasi lebih baik |
| `XGBoost ML Engine` | **DIPERTAHANKAN** | 3-model ensemble tetap dipakai |
| `feature_engine.py` | **DIPERLUAS** | Tambah multi-TF confluence score |
| `risk_manager.py` | **DIPERLUAS** | Tambah dynamic sizing per mode |
| `trade_memory.py` + `post_trade_analyzer.py` | **DIPERTAHANKAN** | Lessons learned loop tetap aktif |

---

## 9. Perbandingan v3.1 vs v4.0

| Aspek | v3.1 | v4.0 |
|-------|------|------|
| Timeframe | Single primary (5M) | Hierarkis: Weekly/Daily → 4H/1H → 15M/5M |
| Analisis Makro | Tidak ada agent khusus | `MacroAgent` dedicated + confirmation rules |
| Mode Operasi | Satu mode (aggressive) | 4 mode dinamis: Opportunistic, Aligned, Conviction, Counter-trend |
| LLM Stack | llama3.1 + qwen2.5:3b + llama3.2:3b (paralel) | Qwen2.5:7b + Mistral:7b + DeepSeek R1:7b (sequential) |
| LLM Supervisor | Tidak ada | DeepSeek R1 dengan chain-of-thought |
| Diversitas LLM | 2 dari keluarga Llama (bias) | 3 keluarga berbeda |
| TP Strategy | Fixed 0.25% | Dinamis: 0.20% / 0.40% / ATR-based |
| Auto-Flip | Tidak ada | ✅ Otomatis setelah close counter-trend |
| VRAM Management | Paralel (~14GB needed) | Sequential (max ~5GB per model) |
| Loop Architecture | Single loop | Fast + Slow + Trigger |
| Sizing | Kelly flat | Kelly dinamis per mode |
| Dashboard | Streamlit (ada) | *(Fase berikutnya)* |

---

## 10. Rencana Pengembangan

### 10.1 Fase

| Fase | Durasi | Target | Deliverable |
|------|--------|--------|-------------|
| **Fase 1 — Foundation** | Minggu 1–2 | Refactor agents + loop architecture | MacroAgent, TrendAgent, ScalpAgent, fast/slow/trigger loop berjalan |
| **Fase 2 — LLM Migration** | Minggu 2–3 | Integrasi sequential LLM pipeline | Qwen + Mistral + DeepSeek R1 cycle berjalan, JSON valid, cache aktif |
| **Fase 3 — Chief Manager** | Minggu 3–4 | Implementasi logic opportunistic + auto-flip + dynamic sizing | 4 mode operasi berfungsi, auto-flip tested |
| **Fase 4 — Testnet Validation** | Minggu 4–6 | Jalankan di Binance Testnet, kumpulkan 200+ trade | Data performa: win rate, avg PnL, max drawdown, Sharpe ratio |
| **Fase 5 — Optimization** | Minggu 6–8 | Tuning bobot, threshold, prompt berdasarkan data testnet | Parameter dioptimasi, win rate stabil > 55% |
| **Fase 6 — Live** | Setelah Fase 5 | Deploy ke akun live dengan modal kecil | Live trading dengan monitoring ketat |

### 10.2 Prioritas Improvement dari v3.1

| # | Item | Alasan | Prioritas |
|---|------|--------|-----------|
| 1 | Sequential LLM loading | VRAM 8GB tidak cukup untuk 3 model paralel | 🔴 CRITICAL |
| 2 | `MacroAgent` (Daily/Weekly) | v3.1 tidak punya konteks besar — sering salah arah | 🔴 HIGH |
| 3 | `DeepSeek R1` supervisor | Reasoning lebih baik saat konflik sinyal | 🔴 HIGH |
| 4 | Auto-flip logic | v3.1 melewatkan momentum setelah close | 🔴 HIGH |
| 5 | Dynamic TP per mode | Fixed 0.25% terlalu kecil saat macro kuat | 🔴 HIGH |
| 6 | Fast + Slow loop separation | LLM delay tidak boleh blokir eksekusi | 🔴 HIGH |
| 7 | Ganti llama3.2 → Mistral:7b | Kurangi bias Llama, tambah diversitas | 🟡 MEDIUM |
| 8 | Upgrade Qwen 3b → 7b | Akurasi teknikal lebih baik | 🟡 MEDIUM |
| 9 | Multi-TF confluence scoring | Sinyal yang align di banyak TF lebih reliable | 🟡 MEDIUM |
| 10 | Telegram alert circuit breaker | Notifikasi darurat real-time | 🟢 LOW |

---

## 11. Risiko & Mitigasi

| Risiko | Dampak | Probabilitas | Mitigasi |
|--------|--------|-------------|----------|
| Leverage 100x — likuidasi mendadak | Kehilangan seluruh margin | Medium | SL ketat 0.13%, circuit breaker 2% daily, sizing konservatif di opportunistic mode |
| LLM cycle 3–5 menit terlalu lambat | Keputusan basi saat pasar cepat | Medium | Fast loop pakai cached decision; trigger loop untuk event mendadak |
| DeepSeek R1 output JSON tidak valid | Supervisor gagal memberi keputusan | Low-Medium | Fallback: retry 1x → cached decision → rule-based composite |
| Overfitting ML ke data sintetis | Performa buruk di live market | Medium | 50% data asli + 50% augmented; walk-forward backtest wajib |
| Thermal throttling laptop | LLM cycle lebih lambat | Medium | Cooldown 5 menit antar cycle; limit thread Ollama (`num_thread`) |
| Internet disconnect saat posisi terbuka | Posisi tidak termonitor | Low | Auto-close jika tidak ada data baru > 2 menit |
| Binance API rate limit | Order gagal terkirim | Low | Retry 3x dengan exponential backoff; WebSocket untuk data real-time |
| False macro confirmation | Entry besar di arah salah | Medium | Require minimal 3 dari 6 konfirmasi; SL tetap aktif di semua mode |

---

## 12. Konfigurasi Parameter

### config.py — Parameter Lengkap

```python
# ── Trading ──────────────────────────────────────────────────
SYMBOL                   = "BTC/USDT:USDT"
LEVERAGE                 = 100
ORDER_TYPE               = "limit"

# ── Signal Thresholds ────────────────────────────────────────
CONF_THRESHOLD_OPEN      = 0.55    # Min composite score untuk open posisi baru
CONF_THRESHOLD_HOLD      = 0.52    # Min score untuk pertahankan posisi aktif
CONF_THRESHOLD_CONVICTION = 0.75   # Score untuk masuk conviction mode

# ── Macro Confirmation ───────────────────────────────────────
MACRO_CONFIRM_MIN_COUNT  = 3       # Minimal 3 dari 6 kondisi terpenuhi
MACRO_CONFIRM_HIGH_REQ   = 1       # Minimal 1 kondisi berbobot TINGGI

# ── Signal Weights ───────────────────────────────────────────
WEIGHT_ML                = 0.40
WEIGHT_TECHNICAL         = 0.20
WEIGHT_LLM_SUPERVISOR    = 0.20
WEIGHT_SENTIMENT         = 0.10
WEIGHT_ORDERFLOW         = 0.10

# ── LLM ──────────────────────────────────────────────────────
LLM_TECHNICAL_MODEL      = "qwen2.5:7b"
LLM_SENTIMENT_MODEL      = "mistral:7b"
LLM_SUPERVISOR_MODEL     = "deepseek-r1:7b"
LLM_OLLAMA_URL           = "http://localhost:11434"
LLM_CYCLE_INTERVAL_MIN   = 5       # Menit antar slow loop
LLM_COOLDOWN_MIN         = 5       # Cooldown setelah satu cycle selesai
LLM_TIMEOUT_SEC          = 120     # Timeout per model
LLM_MAX_TOKENS           = 1024
LLM_CACHE_TTL_MIN        = 15      # Cache expired setelah 15 menit

# ── Risk & Kelly ─────────────────────────────────────────────
MAX_DAILY_LOSS_PCT       = 0.02    # 2%
MAX_DRAWDOWN_PCT         = 0.10    # 10%
KELLY_WINDOW             = 7
KELLY_MAX_FRACTION       = 0.50
MARGIN_MIN_PCT           = 0.01    # 1% (counter-trend minimum)
MARGIN_MAX_PCT           = 0.15    # 15% (conviction mode maximum)
MARGIN_OPPORTUNISTIC_MIN = 0.03    # 3% — minimum agar fee tidak makan profit
MARGIN_OPPORTUNISTIC_MAX = 0.05    # 5%
MARGIN_ALIGNED_MIN       = 0.05    # 5%
MARGIN_ALIGNED_MAX       = 0.08    # 8%
MARGIN_CONVICTION_MIN    = 0.08    # 8%
MARGIN_CONVICTION_MAX    = 0.15    # 15%
MARGIN_COUNTER_TREND_MAX = 0.03    # 3% — counter-trend selalu kecil fixed

# ── TP / SL ──────────────────────────────────────────────────
TP_OPPORTUNISTIC         = 0.0020  # 0.20% harga BTC (~20% ROI di 100x)
TP_ALIGNED               = 0.0040  # 0.40% harga BTC (~40% ROI di 100x)
TP_COUNTER_TREND         = 0.0015  # 0.15% — ambil cepat lalu keluar
SL_OPPORTUNISTIC         = 0.0012  # 0.12%
SL_ALIGNED               = 0.0015  # 0.15%
SL_CONVICTION            = 0.0015  # 0.15% + trailing
SL_COUNTER_TREND         = 0.0010  # 0.10% — paling ketat
TRAILING_ATR_MULT        = 0.50
REVERSAL_MIN_AGE_MIN     = 5
REVERSAL_BYPASS_CONF     = 0.85
FEE_GUARDRAIL_MULT       = 1.50    # Batalkan jika fee > 150% SL risk
FEE_RATIO_MAX            = 0.25    # Batalkan jika fee > 25% expected profit

# ── Loop Timing ──────────────────────────────────────────────
FAST_LOOP_INTERVAL_SEC   = 5
SLOW_LOOP_INTERVAL_MIN   = 5
TRAINING_DAYS            = 90      # Hari data historis untuk training ML

# ── Misc ─────────────────────────────────────────────────────
DISCONNECT_TIMEOUT_SEC   = 120     # Auto-close jika tidak ada data
```

---

## 13. Cara Menjalankan

### 13.1 Prerequisites

```bash
# 1. Install Python 3.10+ dan buat venv
python -m venv venv && source venv/bin/activate  # Linux/Mac
# atau: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install dan jalankan Ollama
# Download dari: https://ollama.ai
ollama serve  # Jalankan di terminal terpisah

# 4. Pull semua model LLM
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull deepseek-r1:7b

# 5. Setup .env
cp .env.example .env
# Edit .env: isi BINANCE_API_KEY dan BINANCE_API_SECRET (testnet)
```

### 13.2 Perintah Utama

```bash
# Validasi environment dan koneksi
python validate.py

# Dry-run — monitoring sinyal tanpa eksekusi order
python main.py --dry-run

# Paksa retrain model ML
python main.py --retrain

# Backtest walk-forward 30 hari
python main.py --backtest

# Jalankan bot penuh (testnet)
python main.py

# Jalankan hanya LLM cycle untuk testing
python main.py --test-llm
```

### 13.3 Urutan Startup Internal

```
main.py
  │
  ├── 1. Load config & validate .env
  ├── 2. Koneksi Binance (ccxt) — testnet
  ├── 3. Init SQLite databases (trading.db, trade_memory.db)
  ├── 4. Fetch OHLCV awal semua timeframe (warmup)
  ├── 5. Train / load ML models
  ├── 6. Jalankan LLM cycle pertama (blocking, untuk dapat initial decision)
  ├── 7. Start Trigger Loop (async thread)
  ├── 8. Start Slow Loop (async thread, interval 5 menit)
  └── 9. Start Fast Loop (main thread, interval 5 detik)
```

---

## 14. Glosarium

| Istilah | Definisi |
|---------|----------|
| **Opportunistic Mode** | Mode di mana bot bebas long/short karena belum ada konfirmasi makro |
| **Aligned Mode** | Mode prioritas searah makro; counter-trend masih diizinkan dengan sizing lebih kecil |
| **Conviction Mode** | Mode keyakinan tertinggi — satu arah, sizing maksimal, trailing stop aktif |
| **Auto-Flip** | Menutup posisi counter-trend dan langsung membuka posisi searah macro bias |
| **Macro Bias** | Arah pasar dari MacroAgent (Weekly/Daily) |
| **Confluence Score** | Jumlah indikator/kondisi yang menunjuk ke arah yang sama |
| **Sequential Loading** | Strategi load-run-unload LLM satu per satu untuk hemat VRAM |
| **Chain-of-Thought** | Kemampuan model AI menampilkan proses berpikir sebelum jawaban akhir |
| **Fast Loop** | Loop eksekusi setiap candle (~5 detik), pakai cached LLM decision |
| **Slow Loop** | Loop LLM cycle setiap 5–10 menit |
| **Trigger Loop** | Loop event-driven — aktif saat ada anomali pasar mendadak |
| **Half-Kelly** | 50% dari Kelly Criterion penuh untuk mengurangi risiko volatilitas |
| **Fee Guardrail** | Batalkan trade jika biaya round-trip > 150% dari SL risk |
| **Premium/Discount Zone** | Area harga di atas (premium) atau di bawah (discount) nilai wajar swing range |
| **ATR** | Average True Range — ukuran volatilitas rata-rata |
| **LLM Cache TTL** | Time-to-live cache keputusan LLM; expired setelah 15 menit |

---

*Dokumen ini adalah PRD resmi untuk AI Crypto Futures Trading Bot v4.0.*  
*Fokus: Core trading engine yang presisi — eksekusi dan manajemen posisi yang andal.*  
*Dashboard monitoring dijadwalkan di fase pengembangan berikutnya.*

---
**PRD v4.2 | Opportunistic Hierarchical Swarm | 2026-06-03**