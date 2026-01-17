import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..interfaces import DataSource


class IBKRConnector(DataSource):
    """Interactive Brokers connector - mock mode for demo."""
    
    def __init__(self, mock_mode=True):
        self.mock_mode = mock_mode
        self.connected = False
        
        if not mock_mode:
            try:
                from ib_insync import IB
                self.ib = IB()
                self.ib.connect('127.0.0.1', 7497, clientId=1)
                self.connected = True
            except Exception as e:
                print(f"IBKR connection failed, using mock mode: {e}")
                self.mock_mode = True
    
    def get_price_history(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        if self.mock_mode:
            return self._generate_mock_data(ticker, start_date, end_date)
        
        # Real IBKR logic would go here
        return pd.DataFrame()
    
    def _generate_mock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic mock data for demo purposes."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='B')
        
        np.random.seed(hash(ticker) % 2**32)
        base_price = 100 + np.random.rand() * 200
        
        returns = np.random.randn(len(dates)) * 0.015
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.randn(len(dates)) * 0.005),
            'High': prices * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
            'Low': prices * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates)),
            'Source': 'IBKR_Mock'
        })
        
        return df
