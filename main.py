# main.py
from historical_data import DataFetcher
from portfolio_analysis import PortfolioAnalyzer
from preliminary_analysis import StockAnalyzer

# Define the list of stock symbols
stocks = ['TSLA', 'AMD', 'SOFI', 'WBA', 'MDLZ', 'AAPL', 'NVDA', 'PLTR', 'SNOW', 'AMZN']

# Create instances of the classes
data_fetcher = DataFetcher()
portfolio_analyzer = PortfolioAnalyzer()
stock_analyzer = StockAnalyzer()

# Fetch and save historical data
data_fetcher.fetch_and_save_data(stocks)

# Perform preliminary analysis
stock_analyzer.analyze(stocks)

# Optimize the portfolio
portfolio_analyzer.optimize_portfolio(stocks)