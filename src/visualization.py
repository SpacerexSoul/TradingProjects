"""
Visualization Module
====================
Handles plotting of forecast results.

Author: Krishna
Date: January 2026
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_forecast(train, test, forecast, conf_int, output_path="forecast_vs_actual.png"):
    """
    Generate and save forecast visualization.
    
    Creates a plot showing:
    - Training data (historical)
    - Actual test values
    - ARIMA forecast
    - 95% Confidence intervals
    
    Parameters:
    -----------
    train : pd.Series
        Training data
    test : pd.Series
        Actual test values
    forecast : pd.Series or array
        Predicted values
    conf_int : pd.DataFrame
        Confidence interval bounds (lower, upper columns)
    output_path : str
        Where to save the plot
    """
    print("\n" + "="*50)
    print("GENERATING VISUALIZATION")
    print("="*50)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot training data
    ax.plot(train.index, train.values, 
            label='Training Data', color='blue', alpha=0.7)
    
    # Plot actual test data
    ax.plot(test.index, test.values, 
            label='Actual (Test)', color='green', linewidth=2)
    
    # Plot forecast
    # Make sure forecast has same index as test
    forecast_series = pd.Series(forecast.values if hasattr(forecast, 'values') else forecast, 
                                index=test.index)
    ax.plot(test.index, forecast_series.values, 
            label='ARIMA Forecast', color='red', linestyle='--', linewidth=2)
    
    # Plot confidence intervals
    if conf_int is not None:
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]
        ax.fill_between(test.index, lower, upper, 
                        color='red', alpha=0.2, label='95% Confidence Interval')
    
    # Formatting
    ax.set_title('ARIMA Stock Price Forecast vs Actual', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stock Price ($)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Add vertical line to show train/test split
    split_date = train.index[-1]
    ax.axvline(x=split_date, color='gray', linestyle=':', alpha=0.7)
    ax.annotate('Train/Test Split', xy=(split_date, ax.get_ylim()[1]), 
                xytext=(10, -10), textcoords='offset points', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
