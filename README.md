# AI Crypto Futures Trading Bot — v4.2
**Opportunistic Hierarchical Swarm System**

Platform trading otomatis berbasis AI untuk instrumen BTC/USDT Futures di Binance. Menggunakan paradigma **Opportunistic Hierarchical Swarm** — sistem di mana setiap agent memiliki spesialisasi timeframe dan domain analisis yang jelas, dan beroperasi secara agresif dan fleksibel dalam kondisi pasar apapun.

## 🚀 Fitur Utama
- **Hierarchical Swarm Intelligence:** Analisis multi-timeframe dari Macro (Weekly/Daily) hingga Scalp (15M/5M).
- **Multi-LLM Stack:** 
  - `Qwen 2.5:7b` (Technical Manager)
  - `Mistral:7b` (Sentiment Manager) 
  - `DeepSeek R1:7b` (Chief Supervisor dengan Chain-of-Thought)
- **ML Engine Ensemble:** XGBoost (Conservative, Balanced, Aggressive) untuk presisi area entry.
- **Dynamic Sizing & Risk Management:** Circuit breaker harian, trailing stop berbasis ATR, dan Kelly sizing dinamis per mode pasar.
- **Fast & Slow Loop Architecture:** Memisahkan eksekusi real-time dari latency LLM inferencing.

## 🧠 Arsitektur Agent

Sistem v4 ini tetap menggunakan konsep **Swarm Intelligence**, namun strukturnya diubah menjadi hierarki yang terpusat di folder `core/agents/`:
1. **MacroAgent:** Menentukan tren makro (Weekly/Daily).
2. **TrendAgent:** Mengukur momentum menengah (4H/1H).
3. **ScalpAgent:** Mencari entry timing (15M/5M).
4. **TeknikalAgent:** Menghitung confluence score multi-TF.
5. **SentimenAgent:** Mengambil data Fear & Greed, RSS, Reddit, dll.
6. **OrderflowAgent:** Analisis orderbook, funding rate, dan open interest.

## 💻 Persyaratan Sistem
- Python 3.10+
- Min. 8GB VRAM (untuk menjalankan model LLM lokal bergiliran / sequential)
- Ollama dengan model: `qwen2.5:7b`, `mistral:7b`, `deepseek-r1:7b`

## ⚙️ Setup & Instalasi

1. **Clone repository:**
   ```bash
   git clone <repo-url>
   cd ai-trading
   ```

2. **Setup Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Untuk Windows
   pip install -r requirements.txt
   ```

3. **Konfigurasi Environment:**
   Buat file `.env` di root direktori dengan kredensial API Binance Anda:
   ```env
   BINANCE_API_KEY=your_api_key_here
   BINANCE_SECRET_KEY=your_secret_key_here
   ```

4. **Jalankan Bot:**
   ```bash
   python main.py
   ```

## ⚠️ Peringatan Risiko
Proyek ini dibuat untuk tujuan eksperimental dan dijalankan di Testnet dengan leverage tinggi. Trading futures cryptocurrency memiliki risiko finansial yang sangat tinggi. Selalu gunakan risk management yang ketat.
