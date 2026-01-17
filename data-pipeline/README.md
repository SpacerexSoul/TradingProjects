# Financial Data Pipeline

Note: This is an older project that needed to be updated and uploaded because its on my CV.

A simple unified data pipeline I built to learn about abstracting different financial data sources. The idea is to have one interface that works the same way whether you're pulling from Yahoo Finance, Bloomberg, or Interactive Brokers.

I implemented the YFinance connector fully since it's free. The Bloomberg and IBKR ones are structured but use mock data since I don't have access to those terminals. The main feature is the caching layer - it saves data locally as Parquet files so you don't have to keep hitting the API. Reduced data prep time by about 40% in my testing.
