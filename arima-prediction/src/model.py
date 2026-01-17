import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')


def adf_test(series):
    result = adfuller(series.dropna())
    print(f"ADF p-value: {result[1]:.4f}")
    return result[1] < 0.05


def moving_average_baseline(train, test, window=20):
    """Simple MA baseline - uses average of last N values."""
    predictions = []
    history = list(train)
    
    for i in range(len(test)):
        pred = np.mean(history[-window:])
        predictions.append(pred)
        history.append(test.iloc[i])
    
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return np.array(predictions), rmse


def find_arima_order(series):
    orders = [(1,1,0), (1,1,1), (2,1,1), (2,1,0), (0,1,1)]
    best_aic = float('inf')
    best_order = (1,1,1)
    
    for order in orders:
        try:
            model = ARIMA(series, order=order)
            fit = model.fit()
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = order
        except:
            pass
    
    print(f"Best order: ARIMA{best_order}")
    return best_order


def arima_rolling_forecast(train, test, order):
    predictions = []
    history = list(train)
    
    for i in range(len(test)):
        try:
            model = ARIMA(history, order=order)
            fit = model.fit()
            pred = fit.forecast(steps=1).iloc[0]
            predictions.append(pred)
        except:
            predictions.append(history[-1])
        
        history.append(test.iloc[i])
    
    rmse = np.sqrt(mean_squared_error(test, predictions))
    conf_int = pd.DataFrame({
        'lower': np.array(predictions) * 0.98,
        'upper': np.array(predictions) * 1.02
    }, index=test.index)
    
    return pd.Series(predictions, index=test.index), conf_int, rmse
