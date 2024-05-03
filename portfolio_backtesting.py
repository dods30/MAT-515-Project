# portfolio_backtesting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PortfolioBacktester:
    def __init__(self, portfolio_weights, rebalance_frequency='QE', additional_contribution=0):
        self.portfolio_weights = portfolio_weights
        self.rebalance_frequency = rebalance_frequency
        self.additional_contribution = additional_contribution

    def load_data(self, stocks, start_date, end_date):
        data = {}
        for stock in stocks:
            df = pd.read_csv(f'{stock}_data.csv', index_col='Date', parse_dates=True)
            df = df.loc[start_date:end_date]
            data[stock] = df
        return data

    def calculate_portfolio_returns(self, stock_returns):
        try:
            portfolio_returns = stock_returns.dot(pd.Series(self.portfolio_weights))
        except ValueError:
            # Align the stock_returns DataFrame with the portfolio weights
            aligned_weights = pd.Series(self.portfolio_weights).reindex(stock_returns.columns, fill_value=0)
            portfolio_returns = stock_returns.dot(aligned_weights)
        return portfolio_returns

    def rebalance_portfolio(self, stock_prices):
        # Align stock_prices with portfolio weights
        aligned_weights = pd.Series(self.portfolio_weights).reindex(stock_prices.columns, fill_value=0)
        current_values = stock_prices.mul(aligned_weights, axis=1)
        total_value = current_values.sum().sum()
        self.portfolio_weights = (current_values.sum() + self.additional_contribution) / (total_value + self.additional_contribution)

    def backtest_portfolio(self, stocks, start_date, end_date):
        data = self.load_data(stocks, start_date, end_date)
        stock_returns = pd.DataFrame()
        for stock in stocks:
            stock_returns[stock] = data[stock]['Adj Close'].pct_change()

        portfolio_returns = self.calculate_portfolio_returns(stock_returns)
        cumulative_returns = (1 + portfolio_returns).cumprod()

        rebalance_dates = cumulative_returns.resample(self.rebalance_frequency).last().index
        for date in rebalance_dates:
            stock_prices = {}
            for stock in stocks:
                try:
                    stock_prices[stock] = data[stock].loc[date, 'Adj Close']
                except KeyError:
                    # Handle missing data for the current stock and date
                    closest_date = data[stock].index.asof(date)
                    if pd.notnull(closest_date):
                        stock_prices[stock] = data[stock].loc[closest_date, 'Adj Close']
                    else:
                        # If no close date is found, skip rebalancing for this stock
                        continue
    
            if len(stock_prices) == len(stocks):
                stock_prices = pd.DataFrame(stock_prices, index=[date])
                self.rebalance_portfolio(stock_prices)

        return cumulative_returns

    def plot_portfolio_growth(self, cumulative_returns, filename='portfolio_growth.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns)
        plt.title('Portfolio Growth')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.savefig(filename)
        plt.close()

    def generate_future_returns(self, returns, num_years=5):
        num_days = 252 * num_years
        future_returns = returns.sample(num_days, replace=True)
        future_cumulative_returns = (1 + future_returns).cumprod()
        return future_cumulative_returns

    def plot_future_growth(self, cumulative_returns, num_simulations=1000, num_years=5, filename='future_growth.png'):
        future_returns_list = []
        for _ in range(num_simulations):
            future_returns = self.generate_future_returns(cumulative_returns.pct_change(), num_years)
            future_returns_list.append(future_returns)

        future_returns_df = pd.concat(future_returns_list, axis=1)
        future_returns_mean = future_returns_df.mean(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(future_returns_mean, label='Most Probable Outcome')
        plt.fill_between(future_returns_mean.index, future_returns_df.quantile(0.05, axis=1), future_returns_df.quantile(0.95, axis=1), alpha=0.3, label='90% Confidence Interval')
        plt.title('Simulated Future Portfolio Growth')
        plt.xlabel('Days')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.savefig(filename)
        plt.close()