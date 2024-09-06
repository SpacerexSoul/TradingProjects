import os
import requests
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient

# API keys
alpha_vantage_api_key = 
fred_api_key = 
news_api_key = 

def fetch_data(symbol, function, api_key):
    """Fetch financial data using Alpha Vantage API."""
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data['quarterlyReports'])

def save_data(data, directory, filename):
    """Save data to CSV in specified directory."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    data.to_csv(os.path.join(directory, filename), index=False)

def fetch_fred_data(series_id, api_key):
    """Fetch data from FRED API."""
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json'
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data['observations'])

def fetch_news_sentiment(query, api_key):
    """Fetch news sentiment data using News API."""
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = newsapi.get_everything(q=query)
    return pd.DataFrame(all_articles['articles'])

def organize_financial_data(symbol, financial_vars):
    """Organize and save financial data by variable and statement type."""
    base_dir = f'data/{symbol}/Financial Data'
    financial_statements = {
        'Income Statement': 'INCOME_STATEMENT',
        'Balance Sheet': 'BALANCE_SHEET',
        'Cash Flow': 'CASH_FLOW'
    }

    # Process and organize each financial statement
    for statement_name, function in financial_statements.items():
        data = fetch_data(symbol, function, alpha_vantage_api_key)
        statement_dir = os.path.join(base_dir, statement_name.replace(' ', '_'))
        
        # Save entire statement CSV
        save_data(data, statement_dir, f'{symbol.lower()}_{statement_name.lower().replace(" ", "_")}.csv')
        
        # Save individual variables in further subdirectories
        for var in data.columns.intersection(financial_vars):
            var_dir = os.path.join(statement_dir, var)
            save_data(data[['fiscalDateEnding', var]], var_dir, f'{var}.csv')

def fetch_and_save_price_data(symbol):
    """Fetch historical price data using yfinance and save it with the Date column."""
    price_data = yf.Ticker(symbol).history(period="max")
    price_data.reset_index(inplace=True)  # Converts the Date index into a column
    price_data_dir = f'data/{symbol}/Historical Price Data'
    save_data(price_data, price_data_dir, 'historical_data.csv')

def main():
    symbol = 'AAPL'
    financial_vars = [
        'grossProfit', 'totalRevenue', 'netIncome', 'totalAssets', 'totalLiabilities',
        'operatingIncome', 'ebit', 'ebitda', 'netDebt', 'totalEquity', 'cashAndCashEquivalentsAtCarryingValue',
        'totalOperatingExpenses', 'costOfRevenue', 'interestExpense', 'incomeTaxExpense'
    ]
    organize_financial_data(symbol, financial_vars)
    fetch_and_save_price_data(symbol)

    # Fetch economic indicators
    economic_indicators = {
        'Interest Rates': 'FEDFUNDS',
        'Inflation Rates': 'CPIAUCSL',
        'Employment Data': 'PAYEMS',
        'GDP Growth Rates': 'GDP'
    }
    for indicator, series_id in economic_indicators.items():
        data = fetch_fred_data(series_id, fred_api_key)
        indicator_dir = f'data/Economic Indicators/{indicator.replace(" ", "_")}'
        save_data(data, indicator_dir, f'{indicator.lower().replace(" ", "_")}.csv')

    # Fetch news sentiment data
    news_sentiment = fetch_news_sentiment('stock market', news_api_key)
    news_sentiment_dir = 'data/Sentiment Analysis/News'
    save_data(news_sentiment, news_sentiment_dir, 'news_sentiment.csv')

    print("Data retrieval and analysis complete. Files saved as CSV.")

if __name__ == "__main__":
    main()

# MADE BY KRISHNA/SPACEREXSOUL 
