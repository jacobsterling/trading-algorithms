from yfinance import Ticker
from pandas import DataFrame, Timestamp


def get_option_prices(
    ticker: Ticker,
    expiration_date: Timestamp,
    strike: float,
    option_type: str,
    n_prices_close_to_strike: int = 1,
) -> DataFrame:
    """Fetch option prices close to the given strike price."""
    option_chain = ticker.option_chain(expiration_date.strftime("%Y-%m-%d"))

    # Select the relevant option data based on option type
    if option_type == "call":
        option_data = option_chain.calls
    elif option_type == "put":
        option_data = option_chain.puts
    else:
        raise ValueError("Invalid option_type. Choose 'call' or 'put'.")

    return option_data.loc[
        (option_data["strike"] - strike).abs().argsort()[:n_prices_close_to_strike]
    ]
