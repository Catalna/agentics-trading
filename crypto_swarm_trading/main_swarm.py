"""
main_swarm.py — Swarm Agent Trading System Entry Point

Usage:
  python main_swarm.py              → Live trading (testnet default)
  python main_swarm.py --dry-run    → Signals only, no orders
  python main_swarm.py --dashboard  → Launch Streamlit dashboard
  python main_swarm.py --validate   → Run validation checks
"""

import argparse
import os
import sys
import signal
import ssl
import urllib3
import logging

# SSL bypass untuk Windows
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# path setup
sys.path.insert(0, os.path.dirname(__file__))

from config.swarm_config import SwarmConfig
from crypto_swarm_trading.core.orchestrator import OrchestratorAgent
from crypto_swarm_trading.core.database import DBManager

def main():
    parser = argparse.ArgumentParser(description='Swarm Agent Trading System')
    parser.add_argument('--dry-run', action='store_true', help="Run without placing real orders")
    parser.add_argument('--dashboard', action='store_true', help="Launch Streamlit dashboard")
    parser.add_argument('--validate', action='store_true', help="Run system validation")
    args = parser.parse_args()
    
    if args.dashboard:
        app_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'streamlit_app.py')
        os.system(f'streamlit run {app_path}')
        return
    
    if args.validate:
        try:
            from validate_swarm import run_validation
            run_validation()
        except ImportError:
            print("Validation script not found.")
        return
    
    config = SwarmConfig()
    if args.dry_run:
        config.DRY_RUN = True
        print("Running in DRY-RUN mode.")
    
    # Setup dirs
    base_dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'models_trained'), exist_ok=True)
    
    orchestrator = OrchestratorAgent(config)
    
    def shutdown(signum, frame):
        print('\nShutting down swarm...')
        orchestrator.emergency_shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    orchestrator.initialize_agents()
    orchestrator.run()

if __name__ == '__main__':
    main()
