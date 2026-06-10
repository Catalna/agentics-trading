import re

filepath = "c:/Users/Radith/ai-trading/INDICATORS.md"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Remove sections 10, 11, 13, 14
# Section 10: Stochastic RSI -> ends before Section 11 or 12
# Let's just remove everything between "## 10. Stochastic RSI" and "## 12. VWAP"
content = re.sub(r'## 10\. Stochastic RSI.*?## 12\. VWAP', '## 12. VWAP', content, flags=re.DOTALL)

# Let's remove everything between "## 13. CCI" and "## 15. Volume Analysis"
content = re.sub(r'## 13\. CCI.*?## 15\. Volume Analysis', '## 15. Volume Analysis', content, flags=re.DOTALL)

# Let's fix the TOC by re-writing it entirely
toc_pattern = r'10\. \[Stochastic RSI\].*?15\. \[Volume Analysis\]\(#15-volume-analysis\)'
new_toc_items = """10. [VWAP](#12-vwap-volume-weighted-average-price)
11. [Volume Analysis](#15-volume-analysis)"""
content = re.sub(toc_pattern, new_toc_items, content, flags=re.DOTALL)

# Re-number the sections in the text
content = content.replace('## 12. VWAP', '## 10. VWAP')
content = content.replace('## 15. Volume Analysis', '## 11. Volume Analysis')
content = content.replace('## 16. Order Block Detection', '## 12. Order Block Detection')
content = content.replace('## 17. Swing Structure', '## 13. Swing Structure')
content = content.replace('## 18. Support & Resistance', '## 14. Support & Resistance')
content = content.replace('## 19. Multi-TF Confluence Score', '## 15. Multi-TF Confluence Score')
content = content.replace('## 20. Volatility Regime', '## 16. Volatility Regime')
content = content.replace('## 21. Library & Implementasi', '## 17. Library & Implementasi')

# Also fix the TOC items after 15
content = content.replace('16. [Order Block Detection](#16-order-block-detection)', '12. [Order Block Detection](#16-order-block-detection)')
content = content.replace('17. [Swing Structure](#17-swing-structure-hh-hl-lh-ll)', '13. [Swing Structure](#17-swing-structure-hh-hl-lh-ll)')
content = content.replace('18. [Support & Resistance](#18-support--resistance)', '14. [Support & Resistance](#18-support--resistance)')
content = content.replace('19. [Multi-TF Confluence Score](#19-multi-tf-confluence-score)', '15. [Multi-TF Confluence Score](#19-multi-tf-confluence-score)')
content = content.replace('20. [Volatility Regime](#20-volatility-regime)', '16. [Volatility Regime](#20-volatility-regime)')
content = content.replace('21. [Library & Implementasi](#21-library--implementasi)', '17. [Library & Implementasi](#21-library--implementasi)')

# Remove references to removed scripts
content = content.replace('    ├── ichimoku.py            ← Ichimoku Cloud\n', '')
content = content.replace('                                      Ichimoku, VWAP, CCI, Williams %R', '                                      VWAP')
content = content.replace('    ├── oscillators.py         ← RSI, MACD, Stoch RSI, CCI, Williams %R', '    ├── oscillators.py         ← RSI, MACD')

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated INDICATORS.md")
