"""
VolatilityRegimeAgent — Layer 2 Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class VolatilityRegimeAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("volatility_regime", config)

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        if 'data' not in input_data or 'df_5m' not in input_data['data']:
            return False
        return True

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            df = pd.DataFrame(input_data['data']['df_5m'])
            if len(df) < 30:
                raise ValueError("Insufficient data points for volatility regime (need >=30)")

            close = df['close']
            returns = np.log(close / close.shift(1)).dropna()
            
            # Annualized realized vol (assuming 5m bars, 252 trading days * 24 hours * 12 bars)
            realized_vol = returns.tail(20).std() * np.sqrt(252 * 288) 
            
            # EWMA Vol Fast (span=12) and Slow (span=26)
            vol_series = returns.rolling(20).std() * np.sqrt(252 * 288)
            vol_fast = vol_series.ewm(span=12, adjust=False).mean().iloc[-1]
            vol_slow = vol_series.ewm(span=26, adjust=False).mean().iloc[-1]
            
            vol_trend = "INCREASING" if vol_fast > vol_slow else "DECREASING"
            
            if realized_vol < 0.3:
                regime = "LOW_VOL"
                pos_mult = 1.5
                leverage = 50
            elif realized_vol < 0.6:
                regime = "MEDIUM_VOL"
                pos_mult = 1.0
                leverage = 30
            elif realized_vol < 1.0:
                regime = "HIGH_VOL"
                pos_mult = 0.6
                leverage = 20
            else:
                regime = "EXTREME_VOL"
                pos_mult = 0.3
                leverage = 10
                
            result = {
                'regime': regime,
                'realized_vol': float(realized_vol),
                'vol_fast': float(vol_fast),
                'vol_slow': float(vol_slow),
                'vol_trend': vol_trend,
                'position_multiplier': pos_mult,
                'suggested_leverage': leverage
            }
            
            self.send_message("ALL", "VOLATILITY_DATA", result, priority=3)
            return result
        except Exception as e:
            self.logger.error(f"Volatility regime analysis failed: {e}")
            return None
