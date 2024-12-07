from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Binary, Real
import numpy as np
import pandas as pd
import yfinance as yf

def get_stock_data(tickers, period='1y'):
    """Fetch historical stock data"""
    data = pd.DataFrame()
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)['Close']
        data[ticker] = hist
    return data

def calculate_metrics(returns):
    """Calculate return metrics"""
    mu = returns.mean() * 252  # Annualized returns
    sigma = returns.cov() * 252  # Annualized covariance
    # Calculate skewness and kurtosis
    skew = returns.skew()
    kurt = returns.kurtosis()
    return mu, sigma, skew, kurt

def create_portfolio_cqm(mu, sigma, skew, kurt, risk_aversion=1, 
                        max_drawdown=0.2, max_cvar=0.15, 
                        min_skew=-1.0, max_kurt=3.0):
    """Create CQM model for portfolio optimization"""
    
    n_assets = len(mu)
    
    # Initialize CQM
    cqm = ConstrainedQuadraticModel()
    
    # Decision variables (portfolio weights)
    w = [Real(f'w_{i}', lower_bound=0.0, upper_bound=1.0) for i in range(n_assets)]
    
    # Objective function: Maximize return - risk_aversion * variance
    objective = 0
    
    # Expected return term
    for i in range(n_assets):
        objective += mu[i] * w[i]
    
    # Risk term
    for i in range(n_assets):
        for j in range(n_assets):
            objective -= risk_aversion * sigma[i,j] * w[i] * w[j]
    
    cqm.set_objective(objective)
    
    # Constraints
    
    # 1. Portfolio weights sum to 1
    weight_sum = sum(w[i] for i in range(n_assets))
    cqm.add_constraint(weight_sum == 1, label='sum_to_one')
    
    # 2. Risk constraints
    risk = sum(sigma[i,j] * w[i] * w[j] 
            for i in range(n_assets) 
            for j in range(n_assets))
    cqm.add_constraint(risk <= max_cvar, label='risk_limit')
    
    # 3. Skewness constraint
    skew_constraint = sum(skew[i] * w[i] for i in range(n_assets))
    cqm.add_constraint(skew_constraint >= min_skew, label='skewness')
    
    # 4. Kurtosis constraint
    kurt_constraint = sum(kurt[i] * w[i] for i in range(n_assets))
    cqm.add_constraint(kurt_constraint <= max_kurt, label='kurtosis')
    
    return cqm

def solve_portfolio_optimization():
    # Sample assets
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG']
    
    # Get historical data
    data = get_stock_data(tickers)
    returns = data.pct_change().dropna()
    
    # Calculate metrics
    mu, sigma, skew, kurt = calculate_metrics(returns)
    
    # Create CQM model
    cqm = create_portfolio_cqm(mu, sigma, skew, kurt)
    
    # Initialize sampler
    sampler = LeapHybridCQMSampler()
    
    # Solve the model
    sampleset = sampler.sample_cqm(cqm, time_limit=20)
    
    # Get best solution
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    
    if len(feasible_sampleset) == 0:
        print("No feasible solution found")
        return None
    
    best_sample = feasible_sampleset.first.sample
    
    # Format results
    portfolio = {tickers[i]: best_sample[f'w_{i}'] 
                for i in range(len(tickers))}
    
    # Calculate portfolio metrics
    weights = np.array([portfolio[ticker] for ticker in tickers])
    expected_return = np.dot(weights, mu)
    portfolio_risk = np.dot(weights.T, np.dot(sigma, weights))
    sharpe_ratio = expected_return / np.sqrt(portfolio_risk)
    
    results = {
        'portfolio': portfolio,
        'expected_return': expected_return,
        'portfolio_risk': portfolio_risk,
        'sharpe_ratio': sharpe_ratio
    }
    
    return results

if __name__ == "__main__":
    results = solve_portfolio_optimization()
    
    if results:
        print("\nOptimal Portfolio Allocation:")
        for asset, weight in results['portfolio'].items():
            print(f"{asset}: {weight:.4f}")
            
        print(f"\nExpected Annual Return: {results['expected_return']:.2%}")
        print(f"Portfolio Risk (Volatility): {np.sqrt(results['portfolio_risk']):.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")