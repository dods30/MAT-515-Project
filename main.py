# main.py
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

# Backtest the portfolio
backtester = PortfolioBacktester(best_portfolios['sharpe']['weights'], rebalance_frequency='QE', additional_contribution=1000)
cumulative_returns = backtester.backtest_portfolio(stocks, '2020-01-01', '2024-02-27')
backtester.plot_portfolio_growth(cumulative_returns)

# Simulate future growth
backtester.plot_future_growth(cumulative_returns)