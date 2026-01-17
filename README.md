# ARIMA Stock Price Prediction

Note: This is an older project that needed to be updated and uploaded because its on my CV.

This is a simple project I made to learn about time series forecasting using ARIMA models. I'm not a quant or anything, just wanted to understand how these statistical models work for predicting stock prices.

The code fetches historical stock data (I used Apple), splits it into training and test sets, and then compares two approaches: a basic moving average (just averaging the last few days) versus an ARIMA model which is more sophisticated and looks at patterns in the data. The ARIMA model ended up being about 38% more accurate than the simple moving average, which was pretty cool to see.

I used Python with yfinance for data, statsmodels for ARIMA, and matplotlib for the charts. Run `python main.py` to see the results.
