import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mplfinance as mpf  

# Function to calculate Value at Risk (VaR) and Expected Shortfall (ES) for the given returns
def calculate_var_es(returns, confidence_level=0.95):
    # Sort the returns to compute the percentile
    sorted_returns = returns.sort_values()
    # Calculate VaR at the specified confidence level
    index_var = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns.iloc[index_var]
    # Calculate ES as the average of the worst losses below the VaR threshold
    es = sorted_returns.iloc[:index_var].mean()
    return var, es

# Function to conduct detailed preliminary analysis on a set of stocks
def analyze(stocks):
    combined_returns = pd.DataFrame()
    annualized_volatilities_dict = {}
    stock_stats = []  # List to hold statistics for all stocks

    # Iterate over each stock to perform the analysis
    for stock in stocks:
        # Load the historical data from a CSV file
        data = pd.read_csv(f'{stock}_data.csv', index_col='Date', parse_dates=True)
        # Compute daily returns from the adjusted close prices
        daily_returns = data['Adj Close'].pct_change()
        combined_returns[stock] = daily_returns

        # Print basic statistical measures for each stock
        print(f"Analysis for {stock}:")
        print(f"Mean: {daily_returns.mean()}")
        print(f"Standard Deviation: {daily_returns.std()}")
        print(f"Skewness: {daily_returns.skew()}")
        print(f"Kurtosis: {daily_returns.kurt()}")
        # Calculate statistical measures
        mean_return = daily_returns.mean()
        std_dev = daily_returns.std()
        skewness = daily_returns.skew()
        kurtosis = daily_returns.kurt()
        annualized_volatility = std_dev * np.sqrt(252)
        var, es = calculate_var_es(daily_returns.dropna())
        stats = {
            'Stock': stock,
            'Mean': mean_return,
            'Standard Deviation': std_dev,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'VaR': var,
            'ES': es,
            'Annualized Volatility': annualized_volatility
        }
        stock_stats.append(stats)
        # Prepare figure for subplots
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'Analysis for {stock}', fontsize=16)

        # Daily Returns
        axs[0, 0].plot(daily_returns)
        axs[0, 0].set_title('Daily Returns')
        axs[0, 0].set_xlabel('Date')
        axs[0, 0].set_ylabel('Returns')

        # Holding Period Returns
        holding_period_returns = (1 + daily_returns).cumprod() - 1
        axs[0, 1].plot(holding_period_returns)
        axs[0, 1].set_title('Holding Period Returns')
        axs[0, 1].set_xlabel('Date')
        axs[0, 1].set_ylabel('Cumulative Returns')

        # Historical Volatility (20-day rolling std)
        rolling_std = daily_returns.rolling(window=20).std()
        axs[1, 0].plot(rolling_std)
        axs[1, 0].set_title('Historical Volatility (20-day rolling std)')
        axs[1, 0].set_xlabel('Date')
        axs[1, 0].set_ylabel('Volatility')

        # Bollinger Bands - Plot on Adjusted Close Price
        adj_close = data['Adj Close']
        rolling_mean = adj_close.rolling(window=20).mean()
        rolling_std = adj_close.rolling(window=20).std()
        bollinger_upper = rolling_mean + (rolling_std * 2)
        bollinger_lower = rolling_mean - (rolling_std * 2)
        axs[1, 1].plot(adj_close.index, adj_close, label='Adjusted Close')
        axs[1, 1].plot(bollinger_upper.index, bollinger_upper, label='Upper Band', linestyle='--')
        axs[1, 1].plot(bollinger_lower.index, bollinger_lower, label='Lower Band', linestyle='--')
        axs[1, 1].fill_between(adj_close.index, bollinger_lower, bollinger_upper, alpha=0.1)
        axs[1, 1].set_title(f'{stock} Bollinger Bands')
        axs[1, 1].legend()

        # VaR and ES
        var, es = calculate_var_es(daily_returns.dropna())
        axs[2, 0].hist(daily_returns.dropna(), bins=50, alpha=0.75)
        axs[2, 0].axvline(x=var, color='r', linestyle='--', label=f'VaR (95%): {var:.2%}')
        axs[2, 0].axvline(x=es, color='g', linestyle='--', label=f'ES (95%): {es:.2%}')
        axs[2, 0].set_title(f'Return Distribution with VaR and ES')
        axs[2, 0].set_xlabel('Returns')
        axs[2, 0].set_ylabel('Frequency')
        axs[2, 0].legend()

        # calculate moving averages
        short_rolling = data['Adj Close'].rolling(window=20).mean()
        long_rolling = data['Adj Close'].rolling(window=50).mean()

        #plot the line chart with moving averages
        axs[2,1].plot(data['Adj Close'], label = 'Adjusted Close')
        axs[2,1].plot(short_rolling, label ='20-day MA', linestyle='--')
        axs[2,1].plot(long_rolling, label ='50-day MA', linestyle='-.')
        axs[2,1].set_title(f'{stock} Line Chart with Moving Averages')
        axs[2,1].legend()
        

        # Adjust layout and show the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Compute and store the annualized volatility for each stock
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        annualized_volatilities_dict[stock] = annualized_volatility
    # Convert the list of dictionaries into a DataFrame
    stats_df = pd.DataFrame(stock_stats)
    # Save the DataFrame to a CSV file
    stats_df.to_csv('stock_statistics.csv', index=False)
    
    stock_labels = list(annualized_volatilities_dict.keys())
    volatilities = list(annualized_volatilities_dict.values())
    # Plotting the bar chart
    plt.bar(stock_labels, volatilities, alpha=0.75)
    plt.title("Annualized Volatilities Across Stocks")
    plt.xlabel('Stocks')
    plt.ylabel('Annualized Volatility')
    plt.xticks(rotation=45)
    plt.show()

    # Correlation matrix
    correlations = combined_returns.corr()
    plt.figure(figsize=(10, 7))
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()