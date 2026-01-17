"""
Model Module
============
Contains the ARIMA model implementation and baseline comparison.

This module handles:
- Stationarity testing (ADF test)
- Baseline naive model
- ARIMA parameter optimization
- Model training and prediction

Author: Krishna
Date: January 2026
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import warnings

# Suppress convergence warnings during grid search
warnings.filterwarnings('ignore')


def adf_stationarity_test(series, significance=0.05):
    """
    Perform Augmented Dickey-Fuller test to check stationarity.
    
    The ADF test tests the null hypothesis that a unit root is present.
    If p-value < significance level, we reject null hypothesis = series is stationary.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to test
    significance : float
        Significance level for the test
        
    Returns:
    --------
    tuple
        (is_stationary, p_value, suggested_d)
    """
    print("\n" + "="*50)
    print("STATIONARITY TEST (Augmented Dickey-Fuller)")
    print("="*50)
    
    result = adfuller(series.dropna())
    
    adf_stat = result[0]
    p_value = result[1]
    
    print(f"ADF Statistic: {adf_stat:.4f}")
    print(f"P-Value: {p_value:.6f}")
    print("Critical Values:")
    for key, val in result[4].items():
        print(f"  {key}: {val:.4f}")
    
    is_stationary = p_value < significance
    
    if is_stationary:
        print(f"\n✓ Series is STATIONARY (p={p_value:.4f} < {significance})")
        suggested_d = 0
    else:
        print(f"\n✗ Series is NOT stationary (p={p_value:.4f} >= {significance})")
        print("  Differencing may be needed (d >= 1)")
        suggested_d = 1
    
    return is_stationary, p_value, suggested_d


def naive_baseline_model(train, test):
    """
    Naive persistence model - predicts that tomorrow's price = today's price.
    
    This is our benchmark. If ARIMA can't beat this, it's not useful.
    
    Parameters:
    -----------
    train : pd.Series
        Training data
    test : pd.Series
        Test data
        
    Returns:
    --------
    tuple
        (predictions, rmse)
    """
    print("\n" + "="*50)
    print("BASELINE MODEL (Naive Persistence)")
    print("="*50)
    
    # For persistence model, prediction for t is the value at t-1
    # So we shift the data by 1
    predictions = []
    history = list(train)
    
    for i in range(len(test)):
        # Predict: tomorrow = today
        pred = history[-1]
        predictions.append(pred)
        # Add actual value to history for next prediction
        history.append(test.iloc[i])
    
    predictions = np.array(predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    
    print(f"Baseline RMSE: {rmse:.4f}")
    
    return predictions, rmse


def find_optimal_arima_params(series, max_p=5, max_d=2, max_q=5):
    """
    Find optimal (p, d, q) parameters for ARIMA using auto_arima.
    
    Uses AIC (Akaike Information Criterion) for model selection.
    Lower AIC = better model.
    
    Parameters:
    -----------
    series : pd.Series
        Training time series
    max_p, max_d, max_q : int
        Maximum values for p, d, q in grid search
        
    Returns:
    --------
    tuple
        Optimal (p, d, q) order
    """
    print("\n" + "="*50)
    print("ARIMA PARAMETER OPTIMIZATION")
    print("="*50)
    print("Running auto_arima... (this may take a minute)")
    
    # Use auto_arima from pmdarima - it's like grid search but smarter
    model = auto_arima(
        series,
        start_p=0, max_p=max_p,
        start_q=0, max_q=max_q,
        d=None, max_d=max_d,  # Let it determine d automatically
        seasonal=False,  # Daily stock data doesn't have strong seasonality
        trace=True,  # Print progress
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True  # Faster than grid search
    )
    
    order = model.order
    print(f"\nOptimal order found: ARIMA{order}")
    print(f"AIC: {model.aic():.2f}")
    
    return order


def train_arima_model(train, order, max_retries=3):
    """
    Train ARIMA model with specified order.
    
    Includes retry logic in case of convergence issues.
    
    Parameters:
    -----------
    train : pd.Series
        Training data
    order : tuple
        (p, d, q) order for ARIMA
    max_retries : int
        Number of retries with different settings if convergence fails
        
    Returns:
    --------
    ARIMA model fit result
    """
    print("\n" + "="*50)
    print(f"TRAINING ARIMA{order} MODEL")
    print("="*50)
    
    for attempt in range(max_retries):
        try:
            model = ARIMA(train, order=order)
            fitted = model.fit()
            print(f"Model trained successfully!")
            print(f"AIC: {fitted.aic:.2f}")
            print(f"BIC: {fitted.bic:.2f}")
            return fitted
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                # Try with adjusted order
                p, d, q = order
                if d < 2:
                    order = (p, d + 1, q)
                    print(f"Retrying with adjusted order: ARIMA{order}")
                else:
                    order = (max(0, p - 1), d, max(0, q - 1))
                    print(f"Retrying with simpler order: ARIMA{order}")
    
    raise RuntimeError("Failed to train ARIMA model after multiple attempts")


def forecast_arima(fitted_model, test_length):
    """
    Generate forecasts from trained ARIMA model.
    
    Parameters:
    -----------
    fitted_model : ARIMAResults
        Trained ARIMA model
    test_length : int
        Number of periods to forecast
        
    Returns:
    --------
    tuple
        (forecast, confidence_intervals)
    """
    # Get forecast with confidence intervals
    forecast_result = fitted_model.get_forecast(steps=test_length)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)  # 95% CI
    
    return forecast, conf_int


def calculate_rmse(actual, predicted):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(actual, predicted))


def compare_models(baseline_rmse, arima_rmse):
    """
    Compare ARIMA performance against baseline.
    
    Parameters:
    -----------
    baseline_rmse : float
        RMSE of baseline model
    arima_rmse : float
        RMSE of ARIMA model
        
    Returns:
    --------
    float
        Improvement percentage (positive = ARIMA is better)
    """
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    improvement = ((baseline_rmse - arima_rmse) / baseline_rmse) * 100
    
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"ARIMA RMSE:    {arima_rmse:.4f}")
    print(f"\n{'*' * 40}")
    print(f"Improvement over Baseline: {improvement:.2f}%")
    print(f"{'*' * 40}")
    
    if improvement >= 15:
        print("✓ SUCCESS: ARIMA achieved target improvement of 15%+")
    elif improvement > 0:
        print("△ ARIMA outperforms baseline, but below 15% target")
    else:
        print("✗ ARIMA did not outperform baseline")
    
    return improvement


if __name__ == "__main__":
    # Quick test with sample data
    print("Model module loaded successfully!")
