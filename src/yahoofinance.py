import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_last_30_days_data(ticker: str) -> pd.DataFrame:
    """
    Fetches the last 30 days of daily data for the given ticker from Yahoo Finance.
    
    Args:
      ticker (str): Stock symbol (e.g., "AAPL", "GOOGL").
    
    Returns:
      pd.DataFrame: DataFrame containing the last 30 days of data.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    return data
