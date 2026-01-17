# ARIMA Stock Price Prediction

## Final Year Project - Time Series Forecasting

This project implements an ARIMA (AutoRegressive Integrated Moving Average) model to forecast stock prices. The goal is to demonstrate how statistical time-series methods can outperform simple baseline models for financial prediction.

### What is ARIMA?

ARIMA is a popular statistical method for time series forecasting that combines:
- **AR (AutoRegressive)**: Uses past values to predict future values
- **I (Integrated)**: Differencing to make the series stationary
- **MA (Moving Average)**: Uses past forecast errors

### Project Structure

```
├── main.py              # Main entry point
├── src/
│   ├── data_loader.py   # Fetches stock data from Yahoo Finance
│   ├── model.py         # ARIMA model and baseline
│   └── visualization.py # Plotting functions
├── requirements.txt
└── forecast_vs_actual.png  # Output visualization
```

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```

This will:
1. Download 2 years of AAPL stock data
2. Test for stationarity using ADF test
3. Train a baseline (naive) model and calculate RMSE
4. Tune and train an ARIMA model
5. Compare performance and print improvement %
6. Generate forecast visualization

### Results

The ARIMA model aims to achieve at least 15% improvement over the naive baseline model in terms of RMSE (Root Mean Squared Error).

### References

- Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control.
- Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice.
