# main.py
import pandas as pd
from historical_data import DataFetcher
from portfolio_analysis import PortfolioAnalyzer
from preliminary_analysis import StockAnalyzer
from portfolio_backtesting import PortfolioBacktester

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
best_portfolios = portfolio_analyzer.optimize_portfolio(stocks)

# Get the subset of stocks and their respective weights for the best Sharpe ratio portfolio
best_sharpe_portfolio_stocks = list(best_portfolios['sharpe']['subset'])
best_sharpe_portfolio_weights = list(best_portfolios['sharpe']['weights'].values())

# Backtest the portfolio
backtester = PortfolioBacktester(best_sharpe_portfolio_weights, rebalance_frequency='QE', additional_contribution=1000)

# Determine a valid validation start date
end_date = '2023-12-31'
validation_start_date = pd.to_datetime(end_date) - pd.DateOffset(months=6)
validation_start_date = validation_start_date.strftime('%Y-%m-%d')

cumulative_returns, validation_cumulative_returns, future_cumulative_returns, future_dates = backtester.forecast_and_compare(best_sharpe_portfolio_stocks, '2020-01-01', end_date, validation_start_date)

# Plot portfolio growth
backtester.plot_portfolio_growth(cumulative_returns, filename='portfolio_growth.png')
backtester.plot_validation_growth(validation_cumulative_returns, future_cumulative_returns, future_dates, filename='validation_portfolio_growth.png')
