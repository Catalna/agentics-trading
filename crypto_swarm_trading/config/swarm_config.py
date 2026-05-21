import os
from dataclasses import dataclass, field
from typing import Dict
from dotenv import load_dotenv

# Load .env from parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

@dataclass
class SwarmConfig:
    """Main configuration for swarm system"""
    
    # Exchange settings
    EXCHANGE: str = 'binanceusdm'
    TESTNET: bool = True
    API_KEY: str = os.getenv('BINANCE_API_KEY', '')
    API_SECRET: str = os.getenv('BINANCE_API_SECRET', '')
    DRY_RUN: bool = False
    
    # Trading pairs
    SYMBOL: str = 'BTC/USDT'
    TIMEFRAMES: Dict[str, str] = field(default_factory=lambda: {
        'execution': '5m',
        'trend': '15m',
        'context': '1h',
        'major': '4h'
    })
    
    # Agent weights for consensus
    AGENT_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'ScalpingStrategyAgent': 1.2,
        'SwingStrategyAgent': 0.8,
        'MeanReversionAgent': 1.0,
        'BreakoutAgent': 1.1
    })
    
    # Risk parameters
    MAX_POSITION_SIZE_PCT: float = 0.05  # 5% of equity
    MAX_LEVERAGE: int = 50
    MAX_DAILY_LOSS_PCT: float = 0.02  # 2%
    MAX_DRAWDOWN_PCT: float = 0.10  # 10%
    RISK_PER_TRADE_PCT: float = 0.02
    
    # ML model settings
    ML_MODEL_PATH: str = os.path.join(os.path.dirname(__file__), '..', 'models_trained')
    ML_RETRAIN_INTERVAL_HOURS: int = 24
    ML_ENSEMBLE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'xgboost': 0.6,
        'lightgbm': 0.4
    })
    
    # Database
    DB_PATH: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'swarm_trading.db')
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_FILE: str = os.path.join(os.path.dirname(__file__), '..', 'logs', 'swarm_system.log')
    
    # Thresholds
    MIN_SIGNAL_CONFIDENCE: float = 0.55
    CONSENSUS_THRESHOLD: float = 0.60
    KELLY_FRACTION: float = 0.25
    
    # MessageBus
    MAX_QUEUE_SIZE: int = 5000
    MESSAGE_TTL_SECONDS: int = 30
    MAX_HISTORY: int = 10000
    
    # Agent health
    HEARTBEAT_INTERVAL: int = 30
    AGENT_ERROR_THRESHOLD: int = 5
    AGENT_COOLDOWN_SECONDS: int = 120
    
    # Orderbook
    ORDERBOOK_DEPTH: int = 20
    WHALE_THRESHOLD_BTC: float = 5.0
    
    # SL/TP
    TP_PCT: float = 0.0012
    SL_PCT: float = 0.0005
    TRAILING_ATR_MULT: float = 0.30
    TP_ATR_MULT: float = 0.50
    SL_ATR_MULT: float = 0.20
