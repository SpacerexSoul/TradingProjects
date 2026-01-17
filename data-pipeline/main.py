import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.connectors.yfinance_conn import YFinanceConnector
from src.pipeline import DataManager


def main():
    print("\nFinancial Data Pipeline Demo\n")
    
    # Setup
    connector = YFinanceConnector()
    manager = DataManager(connector, cache_dir='data')
    
    tickers = ['SPY', 'GOOGL']
    start = '2023-01-01'
    end = '2024-01-01'
    
    # First run - fetch from API
    print("=" * 40)
    print("FIRST RUN (Fetching from API)")
    print("=" * 40)
    
    t1_start = time.time()
    for ticker in tickers:
        df = manager.get_data(ticker, start, end, use_cache=False)
        print(f"\n{ticker} - {len(df)} rows")
        print(df.head())
    t1_end = time.time()
    first_run_time = t1_end - t1_start
    
    # Second run - from cache
    print("\n" + "=" * 40)
    print("SECOND RUN (Loading from Cache)")
    print("=" * 40)
    
    t2_start = time.time()
    for ticker in tickers:
        df = manager.get_data(ticker, start, end, use_cache=True)
        print(f"\n{ticker} - {len(df)} rows loaded from cache")
    t2_end = time.time()
    second_run_time = t2_end - t2_start
    
    # Results
    print("\n" + "=" * 40)
    print("BENCHMARK RESULTS")
    print("=" * 40)
    print(f"First run (API):   {first_run_time:.3f}s")
    print(f"Second run (Cache): {second_run_time:.3f}s")
    
    if first_run_time > 0:
        efficiency_gain = ((first_run_time - second_run_time) / first_run_time) * 100
        print(f"\nEfficiency Gain: {efficiency_gain:.1f}%")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
