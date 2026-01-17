from abc import ABC, abstractmethod
import pandas as pd


class DataSource(ABC):
    """Base interface for all data connectors."""
    
    @abstractmethod
    def get_price_history(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch price history for a ticker.
        Returns DataFrame with columns: Date, Open, High, Low, Close, Volume, Source
        """
        pass
    
    def _standardize_columns(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Ensure consistent column naming."""
        df = df.reset_index()
        
        # Rename to standard format
        column_map = {
            'date': 'Date', 'Date': 'Date',
            'open': 'Open', 'Open': 'Open',
            'high': 'High', 'High': 'High',
            'low': 'Low', 'Low': 'Low',
            'close': 'Close', 'Close': 'Close',
            'volume': 'Volume', 'Volume': 'Volume'
        }
        df = df.rename(columns=column_map)
        
        # Add source column
        df['Source'] = source_name
        
        # Keep only standard columns
        standard_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Source']
        df = df[[c for c in standard_cols if c in df.columns]]
        
        return df
