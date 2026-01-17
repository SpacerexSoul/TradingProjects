import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(train, test, forecast, conf_int, output_path="forecast_vs_actual.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(train.index, train.values, label='Training', color='blue', alpha=0.6)
    ax.plot(test.index, test.values, label='Actual', color='green', linewidth=2)
    ax.plot(test.index, forecast.values, label='ARIMA Forecast', 
            color='red', linestyle='--', linewidth=2)
    
    if conf_int is not None:
        ax.fill_between(test.index, conf_int['lower'], conf_int['upper'],
                        color='red', alpha=0.15, label='95% CI')
    
    ax.axvline(x=train.index[-1], color='gray', linestyle=':', alpha=0.7)
    ax.set_title('ARIMA Forecast vs Actual')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close()
