"""
StopLossAgent — Layer 4 Dynamic SL/TP
"""

from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent

class StopLossAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("stop_loss", config)
        self.tp_pct = getattr(config, 'TP_PCT', 0.0012)
        self.sl_pct = getattr(config, 'SL_PCT', 0.0005)
        self.trailing_atr_mult = getattr(config, 'TRAILING_ATR_MULT', 0.30)
        self.tp_atr_mult = getattr(config, 'TP_ATR_MULT', 0.50)
        self.sl_atr_mult = getattr(config, 'SL_ATR_MULT', 0.20)
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'signal' in input_data and 'technical' in input_data

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        signal = input_data.get('signal', 'HOLD')
        if signal == 'HOLD':
            return None
            
        tech = input_data.get('technical', {}).get('raw', {})
        price = tech.get('close', 50000.0)
        atr = tech.get('atr', 100.0)
        
        # Decide mode based on ATR relative size. 
        # If ATR is very small, use pct. If normal, use ATR.
        use_atr = (atr / price) > 0.0005 
        
        if signal == 'LONG':
            if use_atr:
                sl = price - (atr * self.sl_atr_mult)
                tp = price + (atr * self.tp_atr_mult)
                trailing = price - (atr * self.trailing_atr_mult)
            else:
                sl = price * (1.0 - self.sl_pct)
                tp = price * (1.0 + self.tp_pct)
                trailing = price * (1.0 - self.sl_pct * 0.8)
        else: # SHORT
            if use_atr:
                sl = price + (atr * self.sl_atr_mult)
                tp = price - (atr * self.tp_atr_mult)
                trailing = price + (atr * self.trailing_atr_mult)
            else:
                sl = price * (1.0 + self.sl_pct)
                tp = price * (1.0 - self.tp_pct)
                trailing = price * (1.0 + self.sl_pct * 0.8)
                
        # Validate max SL distance (e.g. max 3%)
        sl_dist = abs(sl - price) / price
        if sl_dist > 0.03:
            if signal == 'LONG':
                sl = price * 0.97
            else:
                sl = price * 1.03
                
        # Formatting
        prec = 2
        if price < 5: prec = 4
        elif price < 50: prec = 3
        
        sl = round(sl, prec)
        tp = round(tp, prec)
        trailing = round(trailing, prec)
        
        result = {
            'sl_price': sl,
            'tp_price': tp,
            'trailing_sl': trailing,
            'sl_pct': round(sl_dist * 100, 3),
            'tp_pct': round(abs(tp - price) / price * 100, 3),
            'use_atr_mode': use_atr
        }
        
        self.send_message("ALL", "STOP_LOSS_DATA", result, priority=5)
        return result
