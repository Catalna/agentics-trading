# 🐝 Swarm Agent Trading System

Sistem trading kripto generasi berikutnya (Next-Generation) yang dibangun dengan arsitektur **Swarm Intelligence** dan **Multi-Agent System (MAS)**.

> [!IMPORTANT]
> Sistem ini dibangun secara terpisah di dalam folder `crypto_swarm_trading/` dan **TIDAK MENGUBAH** atau memodifikasi EA (Expert Advisor) lama.

---

## 🏗️ Arsitektur 6-Layer

Sistem ini memecah monolitik EA menjadi ~20 agen spesialis yang beroperasi dalam 6 layer hierarkis. Agen berkomunikasi secara asynchronous melalui **Priority Message Bus**.

1. **Layer 1: Data Collection**
   - `MarketDataAgent`: Mengambil OHLCV multi-timeframe (5m, 15m, 1h, 4h).
   - `OrderbookAgent`: Analisis depth, spread, slippage, dan whale walls.
   - `NewsAgent`: Scraping RSS feed kripto dan Fear & Greed Index.
   - `SocialSentimentAgent`: Analisis sentimen dari YouTube Crypto (VADER + TextBlob).

2. **Layer 2: Analysis**
   - `TechnicalAnalysisAgent`: Menghitung EMA, MACD, RSI, ADX, VWAP, Bollinger Bands.
   - `MLPredictionAgent`: XGBoost & LightGBM ensemble untuk probabilitas pergerakan.
   - `SentimentAnalysisAgent`: Agregasi skor sentimen menjadi label `BULLISH/BEARISH`.
   - `VolatilityRegimeAgent`: Deteksi regime pasar (LOW/MEDIUM/HIGH/EXTREME VOL).

3. **Layer 3: Strategy (Sinyal)**
   - `ScalpingStrategyAgent`: Strategi utama untuk timeframe 5m (High Frequency).
   - `SwingStrategyAgent`: Strategi multi-jam (15m - 1h) saat trend kuat.
   - `MeanReversionAgent`: Aktif hanya saat pasar RANGING.
   - `BreakoutAgent`: Menangkap momentum dari Bollinger Bands squeeze.

4. **Layer 4: Risk Management**
   - `PositionSizerAgent`: Menggunakan **Fractional Kelly Criterion** untuk sizing dinamis.
   - `StopLossAgent`: Dynamic SL/TP berdasarkan persen atau ATR.
   - `PortfolioRiskAgent`: Monitor total exposure dan max drawdown.
   - `CircuitBreakerAgent`: Tombol darurat jika max daily loss tercapai.

5. **Layer 5: Execution**
   - `OrderRouterAgent`: Memilih LIMIT vs MARKET order berdasarkan likuiditas dan mengeksekusi via ccxt.
   - `ExecutionOptimizerAgent`: Mencari timing optimal berdasarkan spread dan volume.
   - `SlippageMonitorAgent`: Memantau selisih expected vs actual price (post-trade).

6. **Layer 6: Meta-Control**
   - `OrchestratorAgent`: "Otak" utama. Menginisialisasi semua agen, menjalankan algoritma **Konsensus (Weighted Voting)**, dan mengelola kesehatan sistem (Auto-recover, Auto-disable agent).

---

## 🛠️ Instalasi & Setup

1. **Pastikan berada di environment yang tepat:**
   ```bash
   # Jika menggunakan venv
   .\venv\Scripts\activate
   ```

2. **Install dependensi khusus Swarm:**
   ```bash
   cd crypto_swarm_trading
   pip install -r requirements_swarm.txt
   ```

3. **Konfigurasi Environment:**
   Pastikan file `.env` di folder utama (`../.env`) memiliki:
   ```env
   BINANCE_API_KEY=your_api_key_here
   BINANCE_API_SECRET=your_api_secret_here
   ```

4. **Konfigurasi Sistem:**
   Anda dapat mengubah pengaturan sistem di `config/swarm_config.py`, seperti:
   - `TESTNET = True` (Default aman)
   - `MAX_POSITION_SIZE_PCT = 0.05` (Maks 5% equity per trade)
   - `MAX_DAILY_LOSS_PCT = 0.02` (Maks rugi harian 2%)

---

## 🚀 Cara Menjalankan

Masuk ke direktori `crypto_swarm_trading/` terlebih dahulu:
```bash
cd crypto_swarm_trading
```

Tersedia beberapa mode eksekusi melalui `main_swarm.py`:

**1. Validasi Sistem (Sanity Check)**
Pastikan semua modul bisa dimuat dan tidak ada error sintaks.
```bash
python main_swarm.py --validate
```

**2. Dry Run Mode (Simulasi)**
Sistem berjalan normal, menghasilkan sinyal dan konsensus, namun **TIDAK** menempatkan order ke exchange.
```bash
python main_swarm.py --dry-run
```

**3. Live Trading (Testnet / Real)**
Menjalankan sistem secara penuh. (Pastikan `TESTNET = False` di `swarm_config.py` jika ingin menggunakan uang asli).
```bash
python main_swarm.py
```

**4. Real-time Dashboard**
Sistem dilengkapi dengan UI pemantauan real-time berbasis Streamlit.
```bash
python main_swarm.py --dashboard
# atau
streamlit run dashboard/streamlit_app.py
```
Dashboard akan menampilkan:
- Agent Health Status (Uptime, error rate)
- Latest Consensus Distribution
- Equity Curve (PnL)
- System Events & Recent Trades

---

## 🔧 Fitur Stabilitas & Keandalan

- **Auto-Recovery:** Agen yang gagal akan mencoba restart ulang secara otomatis (`base_agent.py`).
- **Circuit Breaker:** Menghentikan trading otomatis jika sentimen ekstrem atau limit rugi tercapai.
- **Dead-Letter Queue:** Pesan yang gagal diproses oleh bus tidak akan hilang begitu saja.
- **SQLite DB:** Semua event, sinyal, dan performa agen dicatat untuk analisis pasca-trading.
