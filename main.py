"""
ARIMA Stock Price Prediction - Main Entry Point
================================================

This script runs the complete ARIMA forecasting pipeline:
1. Load historical stock data
2. Check for stationarity
3. train baseline model
4. Tune and train ARIMA model
5. Compare performance
6. Generate visualization

Author: Krishna
Date: January 2026

Usage:
    python main.py
"""

import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_stock_data, train_test_split
from src.model import (
    adf_stationarity_test,
    naive_baseline_model,
    find_optimal_arima_params,
    train_arima_model,
    forecast_arima,
    calculate_rmse,
    compare_models
)
from src.visualization import plot_forecast


def main():
    """Run the complete ARIMA forecasting pipeline."""
    
    print("\n" + "="*60)
    print("  ARIMA STOCK PRICE PREDICTION - FINAL YEAR PROJECT")
    print("="*60)
    
    # Configuration
    TICKER = "AAPL"
    YEARS = 2
    TRAIN_RATIO = 0.8
    
    print(f"\nConfiguration:")
    print(f"  Ticker: {TICKER}")
    print(f"  Historical data: {YEARS} years")
    print(f"  Train/Test split: {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}")
    
    # Step 1: Load Data
    print("\n" + "-"*60)
    print("STEP 1: Loading Data")
    print("-"*60)
    data = load_stock_data(TICKER, years=YEARS)
    
    # Step 2: Train/Test Split
    print("\n" + "-"*60)
    print("STEP 2: Splitting Data")
    print("-"*60)
    train, test = train_test_split(data, train_ratio=TRAIN_RATIO)
    
    # Step 3: Stationarity Test
    print("\n" + "-"*60)
    print("STEP 3: Testing Stationarity")
    print("-"*60)
    is_stationary, p_value, suggested_d = adf_stationarity_test(train)
    
    # Step 4: Baseline Model
    print("\n" + "-"*60)
    print("STEP 4: Running Baseline Model")
    print("-"*60)
    baseline_preds, baseline_rmse = naive_baseline_model(train, test)
    
    # Step 5: Find Optimal ARIMA Parameters
    print("\n" + "-"*60)
    print("STEP 5: Finding Optimal ARIMA Parameters")
    print("-"*60)
    optimal_order = find_optimal_arima_params(train)
    
    # Step 6: Train ARIMA Model
    print("\n" + "-"*60)
    print("STEP 6: Training ARIMA Model")
    print("-"*60)
    fitted_model = train_arima_model(train, optimal_order)
    
    # Step 7: Generate Forecasts
    print("\n" + "-"*60)
    print("STEP 7: Generating Forecasts")
    print("-"*60)
    forecast, conf_int = forecast_arima(fitted_model, len(test))
    
    # Set proper index for forecast
    forecast.index = test.index
    conf_int.index = test.index
    
    arima_rmse = calculate_rmse(test, forecast)
    print(f"ARIMA RMSE: {arima_rmse:.4f}")
    
    # Step 8: Compare Models
    print("\n" + "-"*60)
    print("STEP 8: Comparing Models")
    print("-"*60)
    improvement = compare_models(baseline_rmse, arima_rmse)
    
    # Step 9: Generate Visualization
    print("\n" + "-"*60)
    print("STEP 9: Generating Visualization")
    print("-"*60)
    plot_forecast(train, test, forecast, conf_int, "forecast_vs_actual.png")
    
    # Final Summary
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"\nResults Summary:")
    print(f"  • Stock analyzed: {TICKER}")
    print(f"  • ARIMA order: {optimal_order}")
    print(f"  • Baseline RMSE: {baseline_rmse:.4f}")
    print(f"  • ARIMA RMSE: {arima_rmse:.4f}")
    print(f"  • Improvement: {improvement:.2f}%")
    print(f"\nArtifacts generated:")
    print(f"  • forecast_vs_actual.png")
    print("\n" + "="*60 + "\n")
    
    return improvement


if __name__ == "__main__":
    try:
        improvement = main()
        if improvement >= 15:
            print("✓ Success! Target improvement achieved.")
            sys.exit(0)
        else:
            print("△ Completed, but target improvement not met.")
            sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
