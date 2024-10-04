import numpy as np
from pandas import Series
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes(
    S: float, K: float, T: float, r: float, sigma: float, option_type="call"
) -> float:
    """
    Calculate the Black-Scholes option price for a European call or put option.

    Parameters:
    S : float : Current stock price
    K : float : Option strike price
    T : float : Time to maturity (in years)
    r : float : Risk-free interest rate (annualized)
    sigma : float : Volatility of the underlying asset (annualized)
    option_type : str : 'call' for a call option, 'put' for a put option

    Returns:
    float : The calculated option price
    """

    d1, d2 = _std(S, K, T, r, sigma)

    match option_type:
        case "call":
            option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        case "put":
            option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        case _:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price


# the standard deviation of the stock price returns, relative to the option strike price.
def _std(S, K, T, r, sigma) -> tuple[float, float]:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


# Delta (Δ) - Managing Directional Risk


# Delta measures the sensitivity of an option's price to changes in the underlying asset's price. For example, a Delta of 0.5 means that the option's price will change by $0.50 for every $1.00 move in the underlying asset.
# Delta Neutrality: To manage directional risk, traders often aim for a delta-neutral portfolio, where the overall Delta of the portfolio is close to zero. This means the portfolio is less sensitive to small moves in the underlying asset.
# Example: If you own options with a net Delta of +50, indicating a bullish bias, you could sell 50 shares of the underlying asset to neutralize your position (assuming 1 option contract corresponds to 100 shares). This hedges the portfolio against small price movements.
# Adjusting Delta: As the underlying asset’s price changes or time passes, the Delta will change, requiring you to adjust your hedge (rebalancing the position). This is known as Delta hedging.
def delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Calculate the Delta of the option."""
    d1, _ = _std(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1
    else:
        raise ValueError("Invalid option_type. Choose 'call' or 'put'.")


# Gamma (Γ) - Managing Delta Risk


# Gamma measures the rate of change of Delta with respect to changes in the underlying asset's price. A high Gamma means that Delta will change rapidly with price movements, making the option’s price more sensitive.
# Gamma and Delta Hedging: Gamma is important for managing the risk of Delta changing too quickly. If your portfolio has a high Gamma, even small moves in the underlying asset can cause large changes in Delta, requiring frequent rebalancing.
# Example: Near expiration, options tend to have high Gamma. If you are Delta-neutral but with high Gamma, a sudden move in the underlying can rapidly shift your Delta, exposing you to significant risk.
# Balancing Gamma and Theta: Often, strategies with high Gamma (like long options) have negative Theta, meaning they lose value as time passes. Conversely, selling options (short positions) typically have positive Theta but negative Gamma, meaning you are exposed to large moves in the underlying asset. Balancing these two is key to risk management.
def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate the Gamma of the option."""
    d1, _ = _std(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


# Theta (Θ) - Managing Time Decay Risk


# Theta measures the sensitivity of an option’s price to the passage of time. It represents the amount of money an option loses per day as it gets closer to expiration, assuming other factors remain constant.
# Using Theta in Strategies: If you have a positive Theta position (typically from selling options), your position benefits from time decay, as options lose value over time. Conversely, a negative Theta position (buying options) loses value as time passes.
# Example: If you expect the market to remain relatively stable, you might sell options (e.g., covered calls or cash-secured puts) to collect premium as Theta decays.
# Theta and Expiration: Theta increases as the option nears expiration. Managing a portfolio’s Theta helps in timing your exit or adjustments, particularly in high Theta decay periods (close to expiration).
def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate the Vega of the option."""
    d1, _ = _std(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


# Vega (ν) - Managing Volatility Risk


# Vega measures the sensitivity of an option’s price to changes in the volatility of the underlying asset. A high Vega means the option is more sensitive to changes in volatility.
# Volatility Trading: If you anticipate a significant change in volatility, you can adjust your portfolio's Vega to benefit from this change.
# Example: If you expect an increase in volatility (e.g., before earnings reports or economic events), you might increase your Vega by buying options, as their value typically increases with higher volatility. Conversely, if you expect a decrease in volatility, you might reduce your Vega by selling options.
# Vega Neutral Strategies: In a Vega-neutral strategy, the portfolio is structured to be insensitive to volatility changes. This can be useful when volatility is expected to remain stable.
def theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Calculate the Theta of the option."""
    d1, d2 = _std(S, K, T, r, sigma)
    if option_type == "call":
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(
            -r * T
        ) * norm.cdf(d2)
    elif option_type == "put":
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(
            -r * T
        ) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option_type. Choose 'call' or 'put'.")
    return theta


# Rho (ρ) - Managing Interest Rate Risk


# Rho measures the sensitivity of an option’s price to changes in the risk-free interest rate. This is generally less impactful for short-term options but can be significant for long-dated options.
# Interest Rate Considerations: If you expect changes in interest rates, you might adjust your portfolio’s Rho. For example, if you expect interest rates to rise, call options (with positive Rho) might increase in value, while put options (with negative Rho) might decrease.
# Example: In a rising interest rate environment, you might prefer holding call options or reduce exposure to long-term puts.
def rho(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Calculate the Rho of the option."""
    _, d2 = _std(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option_type. Choose 'call' or 'put'.")


def black_scholes_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type="call"
):
    """
    Calculate the Greeks for a European call or put option.

    Parameters:
    S : float : Current stock price
    K : float : Option strike price
    T : float : Time to maturity (in years)
    r : float : Risk-free interest rate (annualized)
    sigma : float : Volatility of the underlying asset (annualized)
    option_type : str : 'call' for a call option, 'put' for a put option

    Returns:
    dict : A dictionary containing the values of Delta, Gamma, Theta, Vega, and Rho.
    """
    return Series(
        {
            "Delta": delta(S, K, T, r, sigma, option_type),
            "Gamma": gamma(S, K, T, r, sigma),
            "Theta": theta(S, K, T, r, sigma, option_type),
            "Vega": vega(S, K, T, r, sigma),
            "Rho": rho(S, K, T, r, sigma, option_type),
        }
    )


def implied_volatility(
    option_price: float, S: float, K: float, T: float, r: float, option_type="call"
) -> float:
    """
    Calculate the implied volatility for a given market price of an option.

    Parameters:
    option_price : float : The market price of the option
    S : float : Current stock price
    K : float : Option strike price
    T : float : Time to maturity (in years)
    r : float : Risk-free interest rate (annualized)
    option_type : str : 'call' for a call option, 'put' for a put option

    Returns:
    float : The implied volatility
    """

    def objective_function(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - option_price

    implied_vol = brentq(objective_function, 1e-6, 2.0)  # search in range [0, 2]
    return implied_vol
