#portfolio_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import cm
from itertools import combinations

class PortfolioAnalyzer:
    def __init__(self, risk_free_rate=0.05):
        self.risk_free_rate = risk_free_rate

    def portfolio_statistics(self, weights, mean_returns, covariance_matrix):
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def maximize_sharpe_ratio(self, weights, mean_returns, covariance_matrix):
        return -self.portfolio_statistics(weights, mean_returns, covariance_matrix)[2]

    def minimize_volatility(self, weights, mean_returns, covariance_matrix):
        return self.portfolio_statistics(weights, mean_returns, covariance_matrix)[1]

    def capital_market_line(self, x, sharpe_ratio_tangency):
        return self.risk_free_rate + sharpe_ratio_tangency * x

    def evaluate_subsets(self, stocks, combined_returns, max_assets=None):
        max_assets = max_assets or len(stocks)
        best_sharpe_ratio = -np.inf
        best_volatility = np.inf
        best_sharpe_details = None
        best_vol_details = None
        all_sharpes = []
        all_weights = []
        all_returns = []
        all_vols = []

        for r in range(1, max_assets + 1):
            for subset in combinations(stocks, r):
                subset_returns = combined_returns[list(subset)]
                mean_returns_subset = subset_returns.mean() * 252
                covariance_matrix_subset = subset_returns.cov() * 252
                num_portfolios = 1000

                for i in range(num_portfolios):
                    weights = np.random.random(len(subset))
                    weights /= np.sum(weights)
                    all_weights.append(weights)

                    ret, vol, sharpe = self.portfolio_statistics(weights, mean_returns_subset, covariance_matrix_subset)
                    all_returns.append(ret)
                    all_sharpes.append(sharpe)
                    all_vols.append(vol)

                    if sharpe > best_sharpe_ratio:
                        best_sharpe_ratio = sharpe
                        best_sharpe_details = {
                            'subset': subset,
                            'weights': weights,
                            'return': ret,
                            'volatility': vol,
                            'sharpe_ratio': sharpe
                        }

                    if vol < best_volatility:
                        best_volatility = vol
                        best_vol_details = {
                            'subset': subset,
                            'weights': weights,
                            'return': ret,
                            'volatility': vol,
                            'sharpe_ratio': sharpe
                        }

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

        sorted_portfolios = sorted(zip(all_vols, all_returns), key=lambda x: x[0])
        efficient_vols = []
        efficient_returns = []
        for vol, ret in sorted_portfolios:
            if not efficient_returns or ret > efficient_returns[-1]:
                efficient_vols.append(vol)
                efficient_returns.append(ret)

        sharpe_array = np.array(all_sharpes)
        min_sharpe, max_sharpe = np.min(sharpe_array), np.max(sharpe_array)
        normalize = plt.Normalize(vmin=min_sharpe, vmax=max_sharpe)
        colors = cm.viridis(normalize(sharpe_array))
        plt.figure(figsize=(10, 6))

        
        scatter = plt.scatter(all_vols, all_returns, c=all_sharpes, cmap='viridis', marker='o', alpha=0.3, label='Random Portfolios')
        plt.plot(efficient_vols, efficient_returns, color='black', marker='.', linestyle='-', label='Efficient Frontier')
        plt.scatter(best_sharpe_details['volatility'], best_sharpe_details['return'], c='r', marker='*', s=200, label='Best Sharpe Ratio Portfolio')
        plt.scatter(best_vol_details['volatility'], best_vol_details['return'], c='b', marker='*', s=200, label='Best Minimum Volatility Portfolio')

        sharpe_ratio_tangency = best_sharpe_details['sharpe_ratio']
        volatility_tangency = best_sharpe_details['volatility']
        return_tangency = best_sharpe_details['return']

        vol_range = np.linspace(0, max(all_vols), 100)
        cml_returns = self.capital_market_line(vol_range, sharpe_ratio_tangency)

        plt.plot(vol_range, cml_returns, color='red', linestyle='--', linewidth=2, label='Capital Market Line')
        plt.scatter(0, self.risk_free_rate, marker='o', color='orange', s=100, label='Risk-Free Rate')
        plt.scatter(volatility_tangency, return_tangency, marker='o', color='green', s=100, label='Tangency Portfolio')

        cbar = plt.colorbar(scatter, orientation='vertical')
        cbar.set_label('Sharpe Ratio')
        plt.xlabel('Volatility (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.savefig('efficient_frontier.png')
        plt.close()

        return {
            'sharpe': best_sharpe_details,
            'volatility': best_vol_details
        }

    def optimize_portfolio(self, stocks):
        combined_returns = pd.DataFrame()
        for stock in stocks:
            data = pd.read_csv(f'{stock}_data.csv', index_col='Date', parse_dates=True)
            combined_returns[stock] = data['Adj Close'].pct_change().fillna(0)

        best_portfolios = self.evaluate_subsets(stocks, combined_returns, max_assets=4)

        best_portfolios['sharpe']['weights'] = dict(zip(best_portfolios['sharpe']['subset'], best_portfolios['sharpe']['weights']))
        best_portfolios['volatility']['weights'] = dict(zip(best_portfolios['volatility']['subset'], best_portfolios['volatility']['weights']))

        df_sharpe = pd.DataFrame.from_records([best_portfolios['sharpe']])
        df_volatility = pd.DataFrame.from_records([best_portfolios['volatility']])

        df_sharpe.to_csv('best_sharpe_portfolio.csv', index=False)
        df_volatility.to_csv('best_volatility_portfolio.csv', index=False)

        return best_portfolios