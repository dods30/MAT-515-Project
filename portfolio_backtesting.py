import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from garch_modeling import GARCHModeler

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
        aligned_weights = pd.Series(self.portfolio_weights).reindex(stock_returns.columns, fill_value=0)
        portfolio_returns = stock_returns.dot(aligned_weights)
        return portfolio_returns

    def rebalance_portfolio(self, stock_prices):
        aligned_weights = pd.Series(self.portfolio_weights).reindex(stock_prices.columns, fill_value=0)
        current_values = stock_prices.mul(aligned_weights, axis=1)
        total_value = current_values.sum().sum()
        self.portfolio_weights = (current_values.sum() + self.additional_contribution) / (total_value + self.additional_contribution)

    def backtest_portfolio(self, stocks, start_date, end_date, validation_start_date):
        data = self.load_data(stocks, start_date, end_date)
        stock_returns = pd.DataFrame()
        for stock in stocks:
            stock_returns[stock] = data[stock]['Adj Close'].pct_change()

        portfolio_returns = self.calculate_portfolio_returns(stock_returns).dropna()
        cumulative_returns = (1 + portfolio_returns).cumprod()

        rebalance_dates = cumulative_returns.resample(self.rebalance_frequency).last().index
        for date in rebalance_dates:
            stock_prices = {}
            for stock in stocks:
                try:
                    stock_prices[stock] = data[stock].loc[date, 'Adj Close']
                except KeyError:
                    closest_date = data[stock].index.asof(date)
                    if pd.notnull(closest_date):
                        stock_prices[stock] = data[stock].loc[closest_date, 'Adj Close']
                    else:
                        continue

            if len(stock_prices) == len(stocks):
                stock_prices = pd.DataFrame(stock_prices, index=[date])
                self.rebalance_portfolio(stock_prices)

        # Validation phase
        print(f"Validation Start Date: {validation_start_date}")
        print(f"Available Dates: {portfolio_returns.index}")

        validation_returns = portfolio_returns.loc[validation_start_date:]
        if validation_returns.empty:
            raise ValueError("No data available for the validation period. Check the validation_start_date.")

        validation_cumulative_returns = (1 + validation_returns).cumprod()

        return cumulative_returns, validation_cumulative_returns, stock_returns

    def forecast_and_compare(self, stocks, start_date, end_date, validation_start_date):
        cumulative_returns, validation_cumulative_returns, stock_returns = self.backtest_portfolio(stocks, start_date, end_date, validation_start_date)
        
        print("Cumulative Returns:", cumulative_returns)
        print("Validation Cumulative Returns:", validation_cumulative_returns)
        print("Stock Returns:", stock_returns)

        # Ensure validation_cumulative_returns is not empty before proceeding
        if validation_cumulative_returns.empty:
            raise ValueError("Validation cumulative returns are empty. Ensure there is data for the validation period.")

        # GARCH modeling for future returns
        garch_modeler = GARCHModeler()
        returns_data = [stock_returns[col].dropna() for col in stock_returns.columns if not stock_returns[col].dropna().empty]
        future_returns = garch_modeler.forecast_portfolio_returns(returns_data, self.portfolio_weights, n_periods=252, n_simulations=1000)
        future_cumulative_returns = (1 + future_returns).cumprod(axis=1).mean(axis=0)

        # Generate future dates for plotting
        last_date = validation_cumulative_returns.index[-1]
        future_dates = pd.date_range(start=last_date, periods=len(future_cumulative_returns), freq='B')

        print("Future Returns:", future_returns)
        print("Future Cumulative Returns:", future_cumulative_returns)

        return cumulative_returns, validation_cumulative_returns, future_cumulative_returns, future_dates

    def plot_portfolio_growth(self, cumulative_returns, filename='portfolio_growth.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_returns)
        plt.title('Portfolio Growth')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.savefig(filename)
        plt.close()

    def plot_validation_growth(self, validation_cumulative_returns, future_cumulative_returns, future_dates, filename='validation_growth.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(validation_cumulative_returns.index, validation_cumulative_returns, label='Actual Returns')
        plt.plot(future_dates, future_cumulative_returns, label='GARCH Predicted Returns', linestyle='--')
        plt.title('Validation Portfolio Growth')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.savefig(filename)
        plt.close()
