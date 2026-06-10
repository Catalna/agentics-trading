import re
import os

workspace = "c:/Users/Radith/ai-trading"

# 1. Update INDICATORS.md
ind_path = os.path.join(workspace, "INDICATORS.md")
with open(ind_path, "r", encoding="utf-8") as f:
    ind_content = f.read()

new_toc = """## Daftar Isi

1. KATEGORI TREND
   - [Linear Regression](#linear-regression)
   - [Trend Direction Detection](#trend-direction-detection)
   - [EMA & SMA](#ema--sma)
   - [Swing Structure](#swing-structure)
2. KATEGORI MOMENTUM
   - [RSI](#rsi-relative-strength-index)
   - [MACD](#macd)
3. KATEGORI VOLATILITY
   - [ATR](#atr-average-true-range)
   - [Bollinger Bands](#bollinger-bands)
4. KATEGORI LEVEL
   - [Fibonacci Retracement](#fibonacci-retracement)
   - [Fibonacci Extension](#fibonacci-extension)
   - [Support & Resistance (Pivot)](#support--resistance)
5. KATEGORI VOLUME/FLOW
   - [Volume Analysis](#volume-analysis)
   - [Order Block Detection](#order-block-detection)
   - [VWAP](#vwap-volume-weighted-average-price)
6. AGREGATOR
   - [Multi-TF Confluence Score](#multi-tf-confluence-score)
   - [Volatility Regime](#volatility-regime)
"""
ind_content = re.sub(r'## Daftar Isi.*?---', new_toc + '\n---', ind_content, flags=re.DOTALL)

# Remove obsolete sections (Stochastic, Ichimoku, CCI, Williams)
# We match from "## 10. Stochastic RSI" until "## 15. Volume Analysis"
ind_content = re.sub(r'## 10\. Stochastic RSI.*?## 15\. Volume Analysis', '## Volume Analysis', ind_content, flags=re.DOTALL)

# Replace EMA table logic
new_ema = """### Parameter EMA per Timeframe
- Weekly  → EMA 13 / 144 / 377
- Daily   → EMA 13 / 144 / 377
- 4H      → EMA 13 / 34  / 89
- 1H      → EMA 13 / 34  / 144
- 15M     → EMA 8  / 21  / 55
- 5M      → EMA 5  / 13  / 34
"""
ind_content = re.sub(r'### EMA Alignment Score.*?### Implementasi Python', new_ema + '\n### Implementasi Python', ind_content, flags=re.DOTALL)

with open(ind_path, "w", encoding="utf-8") as f:
    f.write(ind_content)


# 2. Update AGENTS.md
agt_path = os.path.join(workspace, "AGENTS.md")
with open(agt_path, "r", encoding="utf-8") as f:
    agt_content = f.read()

# Replace EMA periods
agt_content = re.sub(r'Periode : EMA 20, EMA 50, EMA 200 \(Daily\)', 'Periode : EMA 13, 144, 377 (Daily)', agt_content)
agt_content = re.sub(r'Periode : EMA 20, 50, 200 di 4H dan 1H', 'Periode : EMA 13, 34, 89 (4H) dan EMA 13, 34, 144 (1H)', agt_content)

# Remove references to Ichimoku, Stochastic, CCI, Williams in TrendAgent and ScalpAgent and TeknikalAgent
agt_content = re.sub(r'#### Ichimoku Cloud.*?#### Premium / Discount Zone', '#### Premium / Discount Zone', agt_content, flags=re.DOTALL)
agt_content = re.sub(r'#### Stochastic RSI.*?#### Order Block Detection', '#### Order Block Detection', agt_content, flags=re.DOTALL)
agt_content = re.sub(r'#### CCI \(Commodity Channel Index\).*?#### Support & Resistance Detection', '#### Support & Resistance Detection', agt_content, flags=re.DOTALL)

with open(agt_path, "w", encoding="utf-8") as f:
    f.write(agt_content)


# 3. Update MODELS.md
mod_path = os.path.join(workspace, "MODELS.md")
with open(mod_path, "r", encoding="utf-8") as f:
    mod_content = f.read()

# Remove Stoch, CCI, Williams, Ichimoku features from code snippets
mod_content = re.sub(r'\s*f"stoch_k_\{tf\}".*?Stochastic %K,', '', mod_content)
mod_content = re.sub(r'\s*f"stoch_d_\{tf\}".*?Stochastic %D,', '', mod_content)
mod_content = re.sub(r'\s*f"cci_\{tf\}".*?CCI 20.*?,', '', mod_content)
mod_content = re.sub(r'\s*f"williams_r_\{tf\}".*?Williams %R,', '', mod_content)
mod_content = re.sub(r'\s*"ichimoku_position".*?,', '', mod_content)
mod_content = re.sub(r'\s*"ichimoku_tk_cross".*?,', '', mod_content)

with open(mod_path, "w", encoding="utf-8") as f:
    f.write(mod_content)


# 4. Update PRD.md
prd_path = os.path.join(workspace, "PRD.md")
with open(prd_path, "r", encoding="utf-8") as f:
    prd_content = f.read()

prd_content = re.sub(r'EMA 9, 20, 50, 200; SMA 20, 50; ADX; Ichimoku Cloud', 'EMA (Custom TF); Linear Regression; Swing Structure', prd_content)
prd_content = re.sub(r'RSI 14; Stochastic RSI; MACD \(12,26,9\); CCI; Williams %R', 'RSI 14; MACD (12,26,9)', prd_content)

with open(prd_path, "w", encoding="utf-8") as f:
    f.write(prd_content)

print("All documentation files updated successfully.")
