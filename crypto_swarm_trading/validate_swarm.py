"""
validate_swarm.py — Sanity check for the Swarm System
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))

def run_validation():
    print("Starting Swarm Validation...\n")
    all_ok = True

    try:
        from config.swarm_config import SwarmConfig
        from crypto_swarm_trading.core.database import DBManager
        from crypto_swarm_trading.core.message_bus import MessageBus
        from crypto_swarm_trading.core.orchestrator import OrchestratorAgent
        print("[OK] Core modules imported.")
    except Exception as e:
        print(f"[FAIL] Core import error: {e}")
        all_ok = False
        traceback.print_exc()

    try:
        from agents.data_collection.market_data_agent import MarketDataAgent
        from agents.analysis.technical_analysis_agent import TechnicalAnalysisAgent
        from agents.strategy.scalping_strategy_agent import ScalpingStrategyAgent
        from agents.risk.circuit_breaker_agent import CircuitBreakerAgent
        from agents.execution.order_router_agent import OrderRouterAgent
        print("[OK] Agent modules imported.")
    except Exception as e:
        print(f"[FAIL] Agent import error: {e}")
        all_ok = False
        traceback.print_exc()

    try:
        from utils.feature_engineering import compute_features
        from models.ensemble_model import EnsembleModel
        print("[OK] Utils & Models imported.")
    except Exception as e:
        print(f"[FAIL] Utils import error: {e}")
        all_ok = False
        traceback.print_exc()

    if all_ok:
        print("\nAll sanity checks passed! The system is structurally sound.")
        sys.exit(0)
    else:
        print("\nValidation failed. Please fix the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    run_validation()
