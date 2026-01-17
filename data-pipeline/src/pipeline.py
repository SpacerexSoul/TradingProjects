import os
import pandas as pd
from datetime import datetime
from .interfaces import DataSource


class DataManager:
    """Handles data fetching, cleaning, and caching."""
    
    def __init__(self, data_source: DataSource, cache_dir: str = 'data'):
        self.data_source = data_source
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_data(self, ticker: str, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{start_date}_{end_date}.parquet")
        
        # Check cache
        if use_cache and os.path.exists(cache_file):
            print(f"Loading {ticker} from cache...")
            df = pd.read_parquet(cache_file)
            return df
        
        # Fetch from source
        print(f"Fetching {ticker} from API...")
        df = self.data_source.get_price_history(ticker, start_date, end_date)
        
        if df.empty:
            return df
        
        # Clean data
        df = self._clean_data(df)
        
        # Save to cache
        df.to_parquet(cache_file, index=False)
        print(f"Saved {ticker} to cache")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data types."""
        # Convert to proper types
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Forward fill missing values
        df = df.ffill()
        
        return df
    
    def get_multiple(self, tickers: list, start_date: str, end_date: str) -> dict:
        """Fetch data for multiple tickers."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_data(ticker, start_date, end_date)
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                results[ticker] = pd.DataFrame()
        return results
