import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Fetch historical data for Apple
apple_data = yf.download('AAPL', start='2023-01-01', end='2024-08-07')

# Use the adjusted close price
apple_data['Date'] = apple_data.index
apple_data.reset_index(drop=True, inplace=True)

# Prepare the data
apple_data = apple_data[['Date', 'Adj Close']]
apple_data.set_index('Date', inplace=True)
apple_data.index = pd.to_datetime(apple_data.index)
apple_data = apple_data.asfreq('B')

# Find the best ARIMA parameters using auto_arima
stepwise_fit = auto_arima(apple_data['Adj Close'], start_p=1, start_q=1,
                          max_p=5, max_q=5, seasonal=False,
                          trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

# Print the best ARIMA parameters
print(stepwise_fit.summary())

# Fit ARIMA model with the best parameters
model = ARIMA(apple_data['Adj Close'], order=stepwise_fit.order)
model_fit = model.fit()

# Predict future prices
future_dates = pd.date_range(start="2024-08-08", end="2024-08-21", freq='B')
future_dates_df = pd.DataFrame(index=future_dates, columns=apple_data.columns)
apple_data = pd.concat([apple_data, future_dates_df])

# Forecast
forecast = model_fit.get_forecast(steps=len(future_dates))
forecast_index = forecast.summary_frame().index
forecast_values = forecast.summary_frame()['mean']

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame({'Date': forecast_index, 'Predicted_Price': forecast_values})

# Display the predictions
print(predictions_df)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(apple_data.index, apple_data['Adj Close'], label='Historical Data')
plt.plot(predictions_df['Date'], predictions_df['Predicted_Price'], label='Predicted Data', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
