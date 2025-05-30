import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes price for a European call option.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate
    sigma : float
        Volatility (annualized)
    
    Returns:
    --------
    float
        Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price


def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate Black-Scholes price for a European put option.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate
    sigma : float
        Volatility (annualized)
    
    Returns:
    --------
    float
        Put option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return put_price


def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes Greeks for European options.
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate
    sigma : float
        Volatility (annualized)
    option_type : str
        'call' or 'put'
    
    Returns:
    --------
    dict
        Dictionary containing all Greeks (delta, gamma, vega, theta, rho)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Common calculations
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    # Gamma (same for calls and puts)
    gamma = n_d1 / (S * sigma * np.sqrt(T))
    
    # Vega (same for calls and puts)
    vega = S * n_d1 * np.sqrt(T) / 100  # Divided by 100 for 1% change
    
    if option_type == 'call':
        delta = N_d1
        theta = (-S * n_d1 * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * N_d2) / 365  # Per day
        rho = K * T * np.exp(-r * T) * N_d2 / 100  # For 1% change
    else:  # put
        delta = N_d1 - 1
        theta = (-S * n_d1 * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365  # Per day
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # For 1% change
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


def implied_volatility(option_price, S, K, T, r, option_type='call', 
                      max_iterations=100, tolerance=1e-6):
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters:
    -----------
    option_price : float
        Market price of the option
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate
    option_type : str
        'call' or 'put'
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance
    
    Returns:
    --------
    float
        Implied volatility
    """
    # Initial guess
    sigma = 0.3
    
    for i in range(max_iterations):
        # Calculate option price with current sigma
        if option_type == 'call':
            price = black_scholes_call(S, K, T, r, sigma)
        else:
            price = black_scholes_put(S, K, T, r, sigma)
        
        # Calculate vega
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        # Newton-Raphson update
        price_diff = price - option_price
        
        if abs(price_diff) < tolerance:
            return sigma
        
        sigma = sigma - price_diff / vega
        
        # Ensure sigma stays positive
        if sigma <= 0:
            sigma = 0.001
    
    # If not converged, return last estimate
    return sigma


# Example usage
if __name__ == "__main__":
    # Example parameters
    S = 100      # Current stock price
    K = 105      # Strike price
    T = 0.25     # 3 months to maturity
    r = 0.05     # 5% risk-free rate
    sigma = 0.2  # 20% volatility
    
    # Calculate option prices
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)
    
    print(f"Black-Scholes Option Prices:")
    print(f"Call Price: ${call_price:.2f}")
    print(f"Put Price: ${put_price:.2f}")
    
    # Calculate Greeks
    call_greeks = black_scholes_greeks(S, K, T, r, sigma, 'call')
    put_greeks = black_scholes_greeks(S, K, T, r, sigma, 'put')
    
    print(f"\nCall Option Greeks:")
    for greek, value in call_greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")
    
    print(f"\nPut Option Greeks:")
    for greek, value in put_greeks.items():
        print(f"{greek.capitalize()}: {value:.4f}")
    
    # Test implied volatility
    test_iv = implied_volatility(call_price, S, K, T, r, 'call')
    print(f"\nImplied Volatility (should be ~{sigma}): {test_iv:.4f}")