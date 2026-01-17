"""
Data Loader Module
==================
Handles fetching historical stock data from Yahoo Finance.

Author: Krishna
Date: January 2026
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def load_stock_data(ticker="AAPL", years=2):
    """
    Fetch historical daily closing prices for a stock.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (default: AAPL)
    years : int
        Number of years of historical data to fetch
        
    Returns:
    --------
    pd.Series
        Daily closing prices with datetime index
    """
    print(f"Fetching {years} years of data for {ticker}...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    # Download data using yfinance
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    
    # Check if we got data
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Extract closing prices
    close_prices = df['Close']
    
    print(f"Successfully loaded {len(close_prices)} data points")
    print(f"Date range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
    
    return close_prices


def train_test_split(data, train_ratio=0.8):
    """
    Split time series data into training and test sets.
    
    Note: For time series, we can't do random splits - need to keep temporal order!
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    train_ratio : float
        Proportion of data for training (default: 0.8)
        
    Returns:
    --------
    tuple
        (train_data, test_data)
    """
    split_idx = int(len(data) * train_ratio)
    train = data[:split_idx]
    test = data[split_idx:]
    
    print(f"Train set: {len(train)} samples")
    print(f"Test set: {len(test)} samples")
    
    return train, test


if __name__ == "__main__":
    # Quick test
    data = load_stock_data("AAPL", years=2)
    train, test = train_test_split(data)
    print(f"\nFirst few values:\n{data.head()}")
