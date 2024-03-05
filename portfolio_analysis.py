import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import cm
from itertools import combinations  

# Define a function to calculate portfolio statistics
def portfolio_statistics(weights, mean_returns, covariance_matrix, risk_free_rate=0.05):
    # Calculate annualized portfolio return based on daily returns, weights, and trading days
    portfolio_return = np.sum(mean_returns * weights) * 252
    # Calculate annualized portfolio volatility using weights and covariance matrix
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights))) * np.sqrt(252)
    # Calculate the Sharpe ratio for the portfolio, adjusting for the risk-free rate
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Define a function to maximize the Sharpe ratio
def maximize_sharpe_ratio(weights, mean_returns, covariance_matrix, risk_free_rate):
    # The function returns the negative Sharpe ratio for minimization algorithm
    return -portfolio_statistics(weights, mean_returns, covariance_matrix, risk_free_rate)[2]

# Define a function to minimize portfolio volatility
def minimize_volatility(weights, mean_returns, covariance_matrix, risk_free_rate=0.05):
    # The function returns portfolio volatility for minimization
    return portfolio_statistics(weights, mean_returns, covariance_matrix, risk_free_rate)[1]

# Calculate returns for the CML
def capital_market_line(x, risk_free_rate, sharpe_ratio_tangency):
    return risk_free_rate + sharpe_ratio_tangency * x

# Define a function to evaluate all subsets of portfolios up to a maximum number of assets
def evaluate_subsets(stocks, combined_returns, max_assets=None, risk_free_rate=0.05):
    # Set default maximum number of assets if not specified
    max_assets = max_assets or len(stocks)
    # Initialize variables to track the best portfolios found
    best_sharpe_ratio = -np.inf
    best_volatility = np.inf
    best_sharpe_details = None
    best_vol_details = None
    # Create lists to store metrics for all generated portfolios
    all_sharpes = []
    all_weights = []
    all_returns = []
    all_vols = []
    # Evaluate portfolio combinations from 1 asset up to the maximum specified
    for r in range(1, max_assets + 1):
        for subset in combinations(stocks, r):
            # Calculate mean returns and covariance matrix for the subset
            subset_returns = combined_returns[list(subset)]
            mean_returns_subset = subset_returns.mean() * 252
            covariance_matrix_subset = subset_returns.cov() * 252
            
            # Generate random portfolios for the current subset
            num_portfolios = 1000
            for i in range(num_portfolios):
                # Generate random weights and normalize them
                weights = np.random.random(len(subset))
                weights /= np.sum(weights)
                all_weights.append(weights)
                

                # Calculate portfolio metrics for the generated weights
                ret, vol, sharpe = portfolio_statistics(weights, mean_returns_subset, covariance_matrix_subset, risk_free_rate)
                all_returns.append(ret)
                all_sharpes.append(sharpe)
                all_vols.append(vol)

                # Update the best portfolios based on Sharpe ratio and volatility
                if sharpe > best_sharpe_ratio:
                    best_sharpe_ratio = sharpe
                    best_sharpe_details = {'subset': subset, 'weights': weights, 'return': ret, 'volatility': vol, 'sharpe_ratio': sharpe}
                
                if vol < best_volatility:
                    best_volatility = vol
                    best_vol_details = {'subset': subset, 'weights': weights, 'return': ret, 'volatility': vol, 'sharpe_ratio': sharpe}


    # Print the details of the best portfolios found
    print("\nBest Sharpe Ratio Portfolio:")
    print("Stocks in Portfolio:", best_sharpe_details['subset'])
    print("Weights:", best_sharpe_details['weights'])
    print("Expected Annual Return:", best_sharpe_details['return'])
    print("Annual Volatility:", best_sharpe_details['volatility'])
    print("Sharpe Ratio:", best_sharpe_details['sharpe_ratio'])

    print("\nBest Minimum Volatility Portfolio:")
    print("Stocks in Portfolio:", best_vol_details['subset'])
    print("Weights:", best_vol_details['weights'])
    print("Expected Annual Return:", best_vol_details['return'])
    print("Annual Volatility:", best_vol_details['volatility'])
    print("Sharpe Ratio:", best_vol_details['sharpe_ratio'])

    # Plot the efficient frontier and mark the best portfolios
    sorted_portfolios = sorted(zip(all_vols, all_returns), key=lambda x: x[0])
    # Initialize lists to hold the efficient portfolios
    efficient_vols = []
    efficient_returns = []
    # Loop through the sorted portfolios
    for vol, ret in sorted_portfolios:
        if not efficient_returns or ret > efficient_returns[-1]:
            efficient_vols.append(vol)
            efficient_returns.append(ret)
    sharpe_array = np.array(all_sharpes)
    min_sharpe, max_sharpe = np.min(sharpe_array), np.max(sharpe_array)
    normalize = plt.Normalize(vmin=min_sharpe, vmax=max_sharpe)
    colors = cm.viridis(normalize(sharpe_array))
    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(all_vols, all_returns, c=colors, cmap='viridis', marker='o', alpha=0.3, label='Random Portfolios')
    plt.plot(efficient_vols, efficient_returns, color='black', marker='.', linestyle='-', label='Efficient Frontier')
    plt.scatter(best_sharpe_details['volatility'], best_sharpe_details['return'], c='r', marker='*', s=200, label='Best Sharpe Ratio Portfolio')
    plt.scatter(best_vol_details['volatility'], best_vol_details['return'], c='b', marker='*', s=200, label='Best Minimum Volatility Portfolio')

    # Find the tangency portfolio from the efficient frontier
    sharpe_ratio_tangency = best_sharpe_details['sharpe_ratio']
    volatility_tangency = best_sharpe_details['volatility']
    return_tangency = best_sharpe_details['return']

    # Get a range of volatilities for plotting the CML
    vol_range = np.linspace(0, max(all_vols), 100)

    # Calculate the CML returns
    cml_returns = capital_market_line(vol_range, risk_free_rate, sharpe_ratio_tangency)

    # Add the CML to the plot
    plt.plot(vol_range, cml_returns, color='red', linestyle='--', linewidth=2, label='Capital Market Line')

    # Highlight the risk-free rate on the y-axis
    plt.scatter(0, risk_free_rate, marker='o', color='orange', s=100, label='Risk-Free Rate')

    # Highlight the tangency portfolio on the efficient frontier
    plt.scatter(volatility_tangency, return_tangency, marker='o', color='green', s=100, label='Tangency Portfolio')

    cbar = plt.colorbar(scatter, orientation='vertical')
    cbar.set_label('Sharpe Ratio')
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.show()
    return {
        'sharpe': best_sharpe_details,
        'volatility': best_vol_details
    }

# Define the main function to optimize the portfolio based on provided stock symbols
def optimize_portfolio(stocks):
    # Initialize a DataFrame to hold combined returns for all stocks
    combined_returns = pd.DataFrame()
    # Load adjusted close price data, calculate returns, and merge into combined DataFrame
    for stock in stocks:
        data = pd.read_csv(f'{stock}_data.csv', index_col='Date', parse_dates=True)
        combined_returns[stock] = data['Adj Close'].pct_change().fillna(0)

    # Find the best portfolios using evaluate_subsets function with a maximum number of assets
    best_portfolios = evaluate_subsets(stocks, combined_returns, max_assets=4)

    # Save best portfolios to a CSV file
    best_portfolios['sharpe']['weights'] = dict(zip(best_portfolios['sharpe']['subset'], best_portfolios['sharpe']['weights']))
    best_portfolios['volatility']['weights'] = dict(zip(best_portfolios['volatility']['subset'], best_portfolios['volatility']['weights']))
    
    df_sharpe = pd.DataFrame.from_records([best_portfolios['sharpe']])
    df_volatility = pd.DataFrame.from_records([best_portfolios['volatility']])
    
    df_sharpe.to_csv('best_sharpe_portfolio.csv', index=False)
    df_volatility.to_csv('best_volatility_portfolio.csv', index=False)

    return best_portfolios