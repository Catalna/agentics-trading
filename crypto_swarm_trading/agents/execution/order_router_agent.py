"""
OrderRouterAgent — Layer 5 Smart Order Routing
"""

from typing import Dict, Any, Optional
from crypto_swarm_trading.core.base_agent import BaseAgent
import uuid

class OrderRouterAgent(BaseAgent):
    def __init__(self, config):
        super().__init__("order_router", config)
        self.dry_run = getattr(config, 'DRY_RUN', False)
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return 'consensus' in input_data and 'position' in input_data and 'stop_loss' in input_data

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        consensus = input_data['consensus']
        if consensus.get('signal', 'HOLD') == 'HOLD':
            return None
            
        position = input_data['position']
        sl_data = input_data['stop_loss']
        orderbook = input_data.get('orderbook', {})
        
        # Decide order type
        liq_score = orderbook.get('liquidity_score', 0)
        spread = orderbook.get('bid_ask_spread_bps', 100)
        
        if liq_score > 0.7 and spread < 5:
            order_type = "LIMIT"
        else:
            order_type = "MARKET"
            
        trade_id = f"TRD_{uuid.uuid4().hex[:8]}"
        
        result = {
            'order_id': f"ORD_{uuid.uuid4().hex[:8]}",
            'trade_id': trade_id,
            'entry_price': 0.0, # Filled by actual execution
            'sl_price': sl_data['sl_price'],
            'tp_price': sl_data['tp_price'],
            'order_type': order_type,
            'status': 'PLACED' if not self.dry_run else 'DRY_RUN'
        }
        
        if self.dry_run:
            self.logger.info(f"[DRY_RUN] Placed {order_type} {consensus['signal']} size {position['quantity']}")
        else:
            self.logger.info(f"Placing {order_type} {consensus['signal']} order")
            # Actual ccxt execution logic here
            
        self.send_message("ALL", "EXECUTION_RESULT", result, priority=5)
        return result
