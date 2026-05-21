"""
MLPredictionAgent — Layer 2 Analysis
"""

import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
try:
    import xgboost as xgb
except ImportError:
    xgb = None
from crypto_swarm_trading.core.base_agent import BaseAgent
import config

class MLPredictionAgent(BaseAgent):
    FEATURE_COLS = [
        'ema5_dist', 'ema20_dist', 'vwap_dist', 'atr_pct', 'volume_change_pct',
        'rsi_scaled', 'adx_scaled', 'macd_norm', 'rsi_slope', 'macd_slope', 'bb_dist'
    ]

    def __init__(self, config_obj):
        super().__init__("ml_prediction", config_obj)
        self.model_path = getattr(config_obj, 'MODEL_PATH', getattr(config, 'MODEL_PATH', './models/xgb_model.json'))
        self.weights = getattr(config_obj, 'ML_ENSEMBLE_WEIGHTS', {'xgboost': 1.0})
        self.xgb_model = self._load_model('xgb')
        self.lgb_model = None

    def _load_model(self, mtype: str):
        if mtype == 'xgb' and xgb is not None and os.path.exists(self.model_path):
            try:
                model = xgb.XGBClassifier()
                model.load_model(self.model_path)
                return model
            except Exception as e:
                self.logger.error(f"Failed to load XGBoost model: {e}")
        return None

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        if 'normalized' not in input_data:
            return False
        for f in self.FEATURE_COLS:
            if f not in input_data['normalized']:
                return False
        return True

    def process(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        feats = input_data['normalized']
        
        # Validate NaN
        if any(v is None or v != v for v in feats.values()): # NaN check
            self.logger.warning("NaN found in features")
            return self._fallback()

        if self.xgb_model is None:
            return self._fallback()
            
        try:
            from core.feature_engine import FEATURE_COLS
            
            row = []
            for col in FEATURE_COLS:
                row.append(feats.get(col, 0.0))
                
            df = pd.DataFrame([row], columns=FEATURE_COLS)
            proba = self.xgb_model.predict_proba(df)[0]
            
            long_prob = float(proba[getattr(config, 'LABEL_LONG', 0)])
            short_prob = float(proba[getattr(config, 'LABEL_SHORT', 1)])
            
            confidence = max(long_prob, short_prob)
            direction = "LONG" if long_prob > short_prob else "SHORT"
            
            result = {
                'direction': direction,
                'long_prob': long_prob,
                'short_prob': short_prob,
                'confidence': confidence,
                'model_agreement': 1.0,
                'model_count': 1
            }
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return self._fallback()

        self.send_message("ALL", "ML_PREDICTION", result, priority=4)
        return result

    def _fallback(self):
        return {
            'direction': "HOLD",
            'long_prob': 0.5,
            'short_prob': 0.5,
            'confidence': 0.5,
            'model_agreement': 1.0,
            'model_count': 0
        }
