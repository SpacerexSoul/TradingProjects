import yfinance as yf
import pandas as pd
from ..interfaces import DataSource


class YFinanceConnector(DataSource):
    """Yahoo Finance data connector - fully functional."""
    
    def get_price_history(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data for {ticker}")
            
            return self._standardize_columns(df, 'YFinance')
        
        except Exception as e:
            print(f"YFinance error for {ticker}: {e}")
            return pd.DataFrame()
