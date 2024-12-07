from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Binary, Real, QuadraticModel
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
    
    # Convert to numpy arrays
    mu_array = mu.to_numpy()
    sigma_array = sigma.to_numpy()
    skew_array = skew.to_numpy()
    kurt_array = kurt.to_numpy()
    
    return mu_array, sigma_array, skew_array, kurt_array

def create_portfolio_cqm(mu, sigma, skew, kurt, risk_aversion=1, 
                        max_cvar=0.15, min_skew=-1.0, max_kurt=3.0):
    """Create CQM model for portfolio optimization"""
    n_assets = len(mu)
    
    # Initialize CQM
    cqm = ConstrainedQuadraticModel()
    
    # Create quadratic model for objective
    qm = QuadraticModel()
    
    # Decision variables (portfolio weights)
    w = [Real(f'w_{i}', lower_bound=0.0, upper_bound=1.0) for i in range(n_assets)]
    
    # Add variables to quadratic model
    for i in range(n_assets):
        qm.add_variable('REAL', f'w_{i}')
    
    # Expected return term (linear terms)
    for i in range(n_assets):
        qm.add_linear(f'w_{i}', mu[i])
    
    # Risk term (quadratic terms)
    for i in range(n_assets):
        for j in range(n_assets):
            if i <= j:  # Only add upper triangular terms
                coeff = -risk_aversion * sigma[i,j]
                if i == j:
                    qm.add_quadratic(f'w_{i}', f'w_{i}', coeff)
                else:
                    qm.add_quadratic(f'w_{i}', f'w_{j}', 2 * coeff)  # Factor of 2 for off-diagonal terms
    
    # Set objective
    cqm.set_objective(qm)
    
    # Constraints
    
    # 1. Portfolio weights sum to 1
    weight_sum = sum(w[i] for i in range(n_assets))
    cqm.add_constraint(weight_sum == 1, label='sum_to_one')
    
    # 2. Individual weight constraints
    for i in range(n_assets):
        cqm.add_constraint(w[i] >= 0, label=f'weight_lower_{i}')
        cqm.add_constraint(w[i] <= 1, label=f'weight_upper_{i}')
    
    # 3. Risk constraint using quadratic model
    risk_qm = QuadraticModel()
    for i in range(n_assets):
        risk_qm.add_variable('REAL', f'w_{i}')
        for j in range(n_assets):
            if i <= j:
                if i == j:
                    risk_qm.add_quadratic(f'w_{i}', f'w_{i}', sigma[i,j])
                else:
                    risk_qm.add_quadratic(f'w_{i}', f'w_{j}', 2 * sigma[i,j])
    
    cqm.add_constraint(risk_qm <= max_cvar, label='risk_limit')
    
    # 4. Skewness constraint (linear)
    skew_constraint = sum(skew[i] * w[i] for i in range(n_assets))
    cqm.add_constraint(skew_constraint >= min_skew, label='skewness')
    
    # 5. Kurtosis constraint (linear)
    kurt_constraint = sum(kurt[i] * w[i] for i in range(n_assets))
    cqm.add_constraint(kurt_constraint <= max_kurt, label='kurtosis')
    
    return cqm

def solve_portfolio_optimization():
    # Sample assets
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
              'TSLA', 'NVDA', 'JPM', 'V', 'PG']
    
    print("Fetching stock data...")
    data = get_stock_data(tickers)
    returns = data.pct_change().dropna()
    
    print("Calculating metrics...")
    mu, sigma, skew, kurt = calculate_metrics(returns)
    
    print("Creating optimization model...")
    cqm = create_portfolio_cqm(mu, sigma, skew, kurt)
    
    print("Initializing sampler...")
    sampler = LeapHybridCQMSampler()
    
    print("Solving optimization problem...")
    sampleset = sampler.sample_cqm(cqm, time_limit=20)
    
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
    try:
        print("Starting portfolio optimization...")
        results = solve_portfolio_optimization()
        
        if results:
            print("\nOptimal Portfolio Allocation:")
            for asset, weight in results['portfolio'].items():
                print(f"{asset}: {weight:.4f}")
                
            print(f"\nExpected Annual Return: {results['expected_return']:.2%}")
            print(f"Portfolio Risk (Volatility): {np.sqrt(results['portfolio_risk']):.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")