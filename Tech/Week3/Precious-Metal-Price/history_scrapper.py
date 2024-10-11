# history_scrapper.py
import yfinance as yf
import pandas as pd
from datetime import datetime


def scrap_history(
    ticker: str, start: datetime = None, end: datetime = None
) -> pd.DataFrame:

    data = yf.download(ticker, start=start, end=end)

    return data
