#preliminar_analysis.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mplfinance as mpf

class StockAnalyzer:
    def __init__(self):
        pass

    def load_data(self, stock):
        return pd.read_csv(f'{stock}_data.csv', index_col='Date', parse_dates=True)

    def calculate_statistics(self, daily_returns):
        return {
            'Mean': daily_returns.mean(),
            'Standard Deviation': daily_returns.std(),
            'Skewness': daily_returns.skew(),
            'Kurtosis': daily_returns.kurt(),
            'Annualized Volatility': daily_returns.std() * np.sqrt(252)
        }

    def calculate_var_es(self, returns, confidence_level=0.95):
        sorted_returns = returns.sort_values()
        index_var = int((1 - confidence_level) * len(sorted_returns))
        var = sorted_returns.iloc[index_var]
        es = sorted_returns.iloc[:index_var].mean()
        return var, es

    def plot_data(self, stock, data):
        daily_returns = data['Adj Close'].pct_change()

        fig, axs = plt.subplots(3, 2, figsize=(14, 18))
        fig.suptitle(f'{stock} Analysis', fontsize=16)

        # Plot daily returns
        axs[0, 0].plot(daily_returns)
        axs[0, 0].set_title('Daily Returns')
        axs[0, 0].set_xlabel('Date')
        axs[0, 0].set_ylabel('Returns')

        # Plot holding period returns
        holding_period_returns = (1 + daily_returns).cumprod() - 1
        axs[0, 1].plot(holding_period_returns)
        axs[0, 1].set_title('Holding Period Returns')
        axs[0, 1].set_xlabel('Date')
        axs[0, 1].set_ylabel('Cumulative Returns')

        # Plot historical volatility
        rolling_std = daily_returns.rolling(window=20).std()
        axs[1, 0].plot(rolling_std)
        axs[1, 0].set_title('Historical Volatility (20-day rolling std)')
        axs[1, 0].set_xlabel('Date')
        axs[1, 0].set_ylabel('Volatility')

        # Plot Bollinger Bands
        adj_close = data['Adj Close']
        rolling_mean = adj_close.rolling(window=20).mean()
        rolling_std = adj_close.rolling(window=20).std()
        bollinger_upper = rolling_mean + (rolling_std * 2)
        bollinger_lower = rolling_mean - (rolling_std * 2)
        axs[1, 1].plot(adj_close.index, adj_close, label='Adjusted Close')
        axs[1, 1].plot(bollinger_upper.index, bollinger_upper, label='Upper Band', linestyle='--')
        axs[1, 1].plot(bollinger_lower.index, bollinger_lower, label='Lower Band', linestyle='--')
        axs[1, 1].fill_between(adj_close.index, bollinger_lower, bollinger_upper, alpha=0.1)
        axs[1, 1].set_title('Bollinger Bands')
        axs[1, 1].set_xlabel('Date')
        axs[1, 1].set_ylabel('Price')
        axs[1, 1].legend()

        # Plot VaR and ES
        var, es = self.calculate_var_es(daily_returns.dropna())
        axs[2, 0].hist(daily_returns.dropna(), bins=50, alpha=0.75)
        axs[2, 0].axvline(x=var, color='r', linestyle='--', label=f'VaR (95%): {var:.2%}')
        axs[2, 0].axvline(x=es, color='g', linestyle='--', label=f'ES (95%): {es:.2%}')
        axs[2, 0].set_title('Return Distribution with VaR and ES')
        axs[2, 0].set_xlabel('Returns')
        axs[2, 0].set_ylabel('Frequency')
        axs[2, 0].legend()

        # Plot line chart with moving averages
        short_rolling = data['Adj Close'].rolling(window=20).mean()
        long_rolling = data['Adj Close'].rolling(window=50).mean()
        axs[2, 1].plot(data['Adj Close'], label='Adjusted Close')
        axs[2, 1].plot(short_rolling, label='20-day MA', linestyle='--')
        axs[2, 1].plot(long_rolling, label='50-day MA', linestyle='-.')
        axs[2, 1].set_title('Line Chart with Moving Averages')
        axs[2, 1].set_xlabel('Date')
        axs[2, 1].set_ylabel('Price')
        axs[2, 1].legend()

        plt.tight_layout()
        plt.savefig(f'{stock}_analysis.png')
        plt.close()

    def analyze_stock(self, stock):
        data = self.load_data(stock)
        daily_returns = data['Adj Close'].pct_change()
        stats = self.calculate_statistics(daily_returns.dropna())
        self.plot_data(stock, data)
        return stats

    def analyze(self, stocks):
        results = {}
        combined_returns = pd.DataFrame()
        for stock in stocks:
            data = self.load_data(stock)
            daily_returns = data['Adj Close'].pct_change()
            combined_returns[stock] = daily_returns
            results[stock] = self.analyze_stock(stock)

        # Save statistical measures to CSV
        stats_df = pd.DataFrame.from_dict(results, orient='index')
        stats_df.to_csv('stock_statistics.csv')

        # Plot annualized volatilities
        annualized_volatilities = stats_df['Annualized Volatility']
        plt.figure(figsize=(10, 6))
        plt.bar(annualized_volatilities.index, annualized_volatilities.values, alpha=0.75)
        plt.title("Annualized Volatilities Across Stocks")
        plt.xlabel('Stocks')
        plt.ylabel('Annualized Volatility')
        plt.xticks(rotation=45)
        plt.savefig('annualized_volatilities.png')
        plt.close()

        # Plot correlation matrix
        correlations = combined_returns.corr()
        plt.figure(figsize=(10, 7))
        sns.heatmap(correlations, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig('correlation_matrix.png')
        plt.close()

        return results