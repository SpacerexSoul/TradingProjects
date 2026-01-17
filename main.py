import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_stock_data, train_test_split
from src.model import adf_test, moving_average_baseline, find_arima_order, arima_rolling_forecast
from src.visualization import plot_forecast


def main():
    TICKER = "AAPL"
    
    print(f"\nARIMA Price Prediction - {TICKER}\n")
    
    prices = load_stock_data(TICKER, years=2)
    train, test = train_test_split(prices)
    print(f"Train: {len(train)}, Test: {len(test)}")
    
    adf_test(train)
    
    # Baseline: 5-day moving average
    _, baseline_rmse = moving_average_baseline(train, test, window=5)
    print(f"MA Baseline RMSE: {baseline_rmse:.4f}")
    
    # ARIMA
    order = find_arima_order(train)
    forecast, conf_int, arima_rmse = arima_rolling_forecast(train, test, order)
    print(f"ARIMA RMSE: {arima_rmse:.4f}")
    
    # Improvement
    improvement = ((baseline_rmse - arima_rmse) / baseline_rmse) * 100
    print(f"\nImprovement over Baseline: {improvement:.2f}%")
    
    plot_forecast(train, test, forecast, conf_int)
    print("Done.")


if __name__ == "__main__":
    main()
