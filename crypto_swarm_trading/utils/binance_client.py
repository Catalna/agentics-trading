"""
utils/binance_client.py — ccxt Binance wrapper utilities
"""

import ccxt
import urllib3
import ssl
import pandas as pd

# Bypass SSL for Windows compatibility where certs might fail
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def build_public_exchange() -> ccxt.Exchange:
    return ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        'verify': False
    })

def build_private_exchange(api_key: str, api_secret: str, testnet: bool = True) -> ccxt.Exchange:
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        'verify': False
    })
    if testnet:
        exchange.set_sandbox_mode(True)
    return exchange

def to_dataframe(ohlcv) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df
