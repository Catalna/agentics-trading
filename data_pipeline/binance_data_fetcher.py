import os
import time
import ccxt
import pandas as pd
from datetime import datetime, timedelta

import config

class BinanceDataFetcher:
    """
    Fetches historical OHLCV data from Binance USDT-M Futures.
    Used for creating ML training datasets.
    """
    def __init__(self, symbol: str = "BTC/USDT:USDT"):
        self.symbol = symbol
        # ccxt.binanceusdm is the dedicated USDT-M Futures exchange class
        self.exchange = ccxt.binanceusdm({
            'enableRateLimit': True,
        })
        self.data_dir = config.DATA_DIR if hasattr(config, 'DATA_DIR') else "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_ohlcv(self, timeframe: str, since_ms: int, limit: int = 1000) -> list:
        try:
            return self.exchange.fetch_ohlcv(self.symbol, timeframe, since=since_ms, limit=limit)
        except Exception as e:
            print(f"Error fetching {timeframe} data: {e}")
            time.sleep(2)
            # Retry once
            return self.exchange.fetch_ohlcv(self.symbol, timeframe, since=since_ms, limit=limit)

    def fetch_historical_data(self, timeframe: str, days: int) -> pd.DataFrame:
        """
        Fetches 'days' worth of historical data for a specific timeframe.
        Saves to CSV and returns a DataFrame.
        """
        print(f"Fetching {days} days of {timeframe} data for {self.symbol}...")
        now = self.exchange.milliseconds()
        since = now - (days * 24 * 60 * 60 * 1000)
        
        all_ohlcv = []
        while since < now:
            print(f"  Fetching from {datetime.fromtimestamp(since/1000)}...")
            ohlcv = self.fetch_ohlcv(timeframe, since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            # update since to the last timestamp + 1ms to avoid duplicates
            since = ohlcv[-1][0] + 1
            time.sleep(0.5) # respect rate limit
            
        if not all_ohlcv:
            print(f"Failed to fetch any data for {timeframe}")
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Drop duplicates just in case
        df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Save to CSV
        file_path = os.path.join(self.data_dir, f"raw_{timeframe}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved {len(df)} rows to {file_path}")
        return df

if __name__ == "__main__":
    fetcher = BinanceDataFetcher(symbol="BTC/USDT:USDT")
    # Fetch 30 days of data for multiple timeframes
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    days = 30
    
    for tf in timeframes:
        fetcher.fetch_historical_data(timeframe=tf, days=days)
    
    print("Data fetching complete.")
