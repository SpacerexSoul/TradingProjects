import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_stock_data(ticker="AAPL", years=2):
    print(f"Fetching {ticker}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    
    print(f"Loaded {len(df)} points")
    return df['Close']


def train_test_split(data, train_ratio=0.8):
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]
