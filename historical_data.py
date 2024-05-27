#historical_data.py
import yfinance as yf

class DataFetcher:
    def __init__(self, start_date='2020-01-01', end_date='2024-05-20'):
        self.start_date = start_date
        self.end_date = end_date

    def fetch_and_save_data(self, stocks):
        """
        Fetches historical stock data from Yahoo Finance and saves it to CSV files.

        Args:
            stocks (list): A list of stock symbols to fetch data for.
        """
        for stock in stocks:
            # Download historical data from start date to end date for the given stock
            data = yf.download(stock, start=self.start_date, end=self.end_date)
            # Save the data to a CSV file named after the stock symbol
            data.to_csv(f'{stock}_data.csv')