import pandas as pd
import numpy as np
from ..interfaces import DataSource


class BloombergConnector(DataSource):
    """Bloomberg connector - mock mode for demo."""
    
    def __init__(self, mock_mode=True):
        self.mock_mode = mock_mode
        self.connected = False
        
        if not mock_mode:
            try:
                import blpapi
                # Real Bloomberg connection logic
                self.connected = True
            except Exception as e:
                print(f"Bloomberg unavailable, using mock mode: {e}")
                self.mock_mode = True
    
    def get_price_history(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self.mock_mode:
            return self._generate_mock_data(ticker, start_date, end_date)
        
        return pd.DataFrame()
    
    def _generate_mock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='B')
        
        np.random.seed(hash(ticker + 'bbg') % 2**32)
        base_price = 150 + np.random.rand() * 150
        
        returns = np.random.randn(len(dates)) * 0.012
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.randn(len(dates)) * 0.004),
            'High': prices * (1 + np.abs(np.random.randn(len(dates)) * 0.008)),
            'Low': prices * (1 - np.abs(np.random.randn(len(dates)) * 0.008)),
            'Close': prices,
            'Volume': np.random.randint(500000, 8000000, len(dates)),
            'Source': 'Bloomberg_Mock'
        })
        
        return df
