import numpy as np
from typing import Literal, Tuple


def price_asian_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_simulations: int = 10000,
    n_steps: int = 252,
    option_type: Literal["call", "put"] = "call",
    average_type: Literal["arithmetic", "geometric"] = "arithmetic",
    seed: int = None
) -> Tuple[float, float]:
    """
    Price an Asian option using Monte Carlo simulation.
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    n_simulations : int, default=10000
        Number of Monte Carlo simulations
    n_steps : int, default=252
        Number of time steps (e.g., 252 for daily averaging over a year)
    option_type : str, default="call"
        Type of option - "call" or "put"
    average_type : str, default="arithmetic"
        Type of average - "arithmetic" or "geometric"
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    Tuple[float, float]
        (option_price, standard_error)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Time step
    dt = T / n_steps
    
    # Generate random paths
    Z = np.random.randn(n_simulations, n_steps)
    
    # Initialize price paths
    S = np.zeros((n_simulations, n_steps + 1))
    S[:, 0] = S0
    
    # Generate stock price paths using geometric Brownian motion
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    
    # Calculate average prices (excluding initial price)
    if average_type == "arithmetic":
        avg_prices = np.mean(S[:, 1:], axis=1)
    else:  # geometric
        avg_prices = np.exp(np.mean(np.log(S[:, 1:]), axis=1))
    
    # Calculate payoffs
    if option_type == "call":
        payoffs = np.maximum(avg_prices - K, 0)
    else:  # put
        payoffs = np.maximum(K - avg_prices, 0)
    
    # Discount payoffs to present value
    discounted_payoffs = payoffs * np.exp(-r * T)
    
    # Calculate option price and standard error
    option_price = np.mean(discounted_payoffs)
    standard_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
    
    return option_price, standard_error


def price_asian_option_control_variate(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_simulations: int = 10000,
    n_steps: int = 252,
    option_type: Literal["call", "put"] = "call",
    seed: int = None
) -> Tuple[float, float]:
    """
    Price an arithmetic Asian option using Monte Carlo with control variates.
    Uses geometric Asian option as control variate.
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    n_simulations : int, default=10000
        Number of Monte Carlo simulations
    n_steps : int, default=252
        Number of time steps
    option_type : str, default="call"
        Type of option - "call" or "put"
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    Tuple[float, float]
        (option_price, standard_error)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Time step
    dt = T / n_steps
    
    # Generate random paths
    Z = np.random.randn(n_simulations, n_steps)
    
    # Initialize price paths
    S = np.zeros((n_simulations, n_steps + 1))
    S[:, 0] = S0
    
    # Generate stock price paths
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    
    # Calculate arithmetic and geometric averages
    arith_avg = np.mean(S[:, 1:], axis=1)
    geom_avg = np.exp(np.mean(np.log(S[:, 1:]), axis=1))
    
    # Calculate payoffs for both
    if option_type == "call":
        arith_payoffs = np.maximum(arith_avg - K, 0)
        geom_payoffs = np.maximum(geom_avg - K, 0)
    else:
        arith_payoffs = np.maximum(K - arith_avg, 0)
        geom_payoffs = np.maximum(K - geom_avg, 0)
    
    # Discount payoffs
    arith_discounted = arith_payoffs * np.exp(-r * T)
    geom_discounted = geom_payoffs * np.exp(-r * T)
    
    # Calculate the known price of geometric Asian option (closed-form approximation)
    # For simplicity, using Monte Carlo estimate
    geom_price = np.mean(geom_discounted)
    
    # Apply control variate technique
    cov = np.cov(arith_discounted, geom_discounted)[0, 1]
    var_geom = np.var(geom_discounted)
    
    # Optimal coefficient
    c = -cov / var_geom if var_geom > 0 else 0
    
    # Adjusted payoffs
    adjusted_payoffs = arith_discounted + c * (geom_discounted - geom_price)
    
    # Calculate option price and standard error
    option_price = np.mean(adjusted_payoffs)
    standard_error = np.std(adjusted_payoffs) / np.sqrt(n_simulations)
    
    return option_price, standard_error


# Example usage
if __name__ == "__main__":
    # Example parameters
    S0 = 100      # Initial stock price
    K = 105       # Strike price
    T = 1         # 1 year to maturity
    r = 0.05      # 5% risk-free rate
    sigma = 0.2   # 20% volatility
    
    print("Asian Option Pricing Example")
    print("=" * 40)
    print(f"Initial Stock Price: ${S0}")
    print(f"Strike Price: ${K}")
    print(f"Time to Maturity: {T} year")
    print(f"Risk-free Rate: {r*100}%")
    print(f"Volatility: {sigma*100}%")
    print("\nResults:")
    print("-" * 40)
    
    # Arithmetic Asian Call
    price, se = price_asian_option(S0, K, T, r, sigma, n_simulations=50000, option_type="call")
    print(f"Arithmetic Asian Call: ${price:.4f} (SE: ${se:.4f})")
    
    # Geometric Asian Call
    price, se = price_asian_option(S0, K, T, r, sigma, n_simulations=50000, option_type="call", average_type="geometric")
    print(f"Geometric Asian Call: ${price:.4f} (SE: ${se:.4f})")
    
    # Arithmetic Asian Put
    price, se = price_asian_option(S0, K, T, r, sigma, n_simulations=50000, option_type="put")
    print(f"Arithmetic Asian Put: ${price:.4f} (SE: ${se:.4f})")
    
    # With control variate
    price, se = price_asian_option_control_variate(S0, K, T, r, sigma, n_simulations=50000)
    print(f"\nArithmetic Asian Call (Control Variate): ${price:.4f} (SE: ${se:.4f})")
