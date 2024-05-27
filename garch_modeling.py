import numpy as np
from scipy.optimize import minimize

class GARCHModeler:
    def __init__(self):
        pass

    def garch_log_likelihood(self, params, returns):
        omega, alpha, beta = params
        n = len(returns)
        sigma2 = np.zeros(n)
    
        # Initialize the first variance estimate
        sigma2[0] = np.var(returns)
    
        # Compute the conditional variances
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]
    
        # Compute the log-likelihood
        log_likelihood = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (returns**2) / sigma2).sum()
        return -log_likelihood  # We minimize the negative log-likelihood

    def estimate_garch_parameters(self, returns):
        initial_params = [0.01, 0.1, 0.8]  # Initial guesses for omega, alpha, beta
        bounds = [(1e-6, None), (1e-6, 1), (1e-6, 1)]  # Bounds for the parameters
        result = minimize(self.garch_log_likelihood, initial_params, args=(returns,), bounds=bounds, method='L-BFGS-B')
        return result.x

    def simulate_garch_returns(self, returns, n_periods, n_simulations, mean_return=0):
        params = self.estimate_garch_parameters(returns)
        omega, alpha, beta = params
        simulated_returns = np.zeros((n_simulations, n_periods))
        
        for j in range(n_simulations):
            returns_sim = np.zeros(n_periods)
            vol = np.zeros(n_periods)
            vol[0] = np.sqrt(omega / (1 - alpha - beta))
            for t in range(1, n_periods):
                vol[t] = np.sqrt(omega + alpha * returns_sim[t-1]**2 + beta * vol[t-1]**2)
                returns_sim[t] = mean_return + vol[t] * np.random.normal()
            simulated_returns[j, :] = returns_sim
        
        return simulated_returns

    def forecast_portfolio_returns(self, returns_data, weights, n_periods, n_simulations):
        num_stocks = len(weights)
        all_simulated_returns = np.zeros((n_simulations, n_periods, num_stocks))
        
        for i, stock_returns in enumerate(returns_data):
            simulated_returns = self.simulate_garch_returns(stock_returns, n_periods, n_simulations)
            all_simulated_returns[:, :, i] = simulated_returns
        
        portfolio_returns = np.tensordot(all_simulated_returns, weights, axes=((2), (0)))
        return portfolio_returns
