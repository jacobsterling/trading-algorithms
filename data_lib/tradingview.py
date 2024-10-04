from tvDatafeed import TvDatafeed, Interval
from os import getenv
from dotenv import load_dotenv

load_dotenv(".env")

TV = TvDatafeed(
    username=getenv("TV_USERNAME"),
    password=getenv("TV_PASSWORD"),
)


def fetch_yield(ticker: str = "US03MY") -> float:
    """Fetch the latest bond yield from TradingView."""
    try:
        data = TV.get_hist(
            symbol=ticker, exchange="TVC", interval=Interval.in_daily, n_bars=1
        )
        return data["close"].iloc[-1] / 100  # Convert percentage to decimal
    except Exception as e:
        print(f"Error fetching yield for {ticker}: {e}")
        return None


def get_risk_free_rate(maturity_in_years: float) -> float:
    """Fetch the risk-free rate based on maturity in years."""

    # Define ticker mappings for various maturities
    tickers = {
        (1 / 12): "US01MY",  # 1-month bond yield
        (1 / 6): "US02MY",  # 2-month bond yield
        (1 / 4): "US03MY",  # 3-month bond yield
        (1 / 3): "US04MY",  # 4-month bond yield
        (5 / 12): "US05MY",  # 5-month bond yield
        (1 / 2): "US06MY",  # 6-month bond yield
        (7 / 12): "US07MY",  # 7-month bond yield
        (2 / 3): "US08MY",  # 8-month bond yield
        (3 / 4): "US09MY",  # 9-month bond yield
        (5 / 6): "US10MY",  # 10-month bond yield
        (11 / 12): "US1MY",  # 11-month bond yield
        1.0: "US01Y",  # 1-year bond yield
        2.0: "US02Y",  # 2-year bond yield
        3.0: "US03Y",  # 3-year bond yield
        4.0: "US04Y",  # 4-year bond yield
        5.0: "US05Y",  # 5-year bond yield
        10.0: "US10Y",  # 10-year bond yield
        20.0: "US20Y",  # 20-year bond yield
        30.0: "US30Y",  # 30-year bond yield
    }

    # Find the closest ticker based on maturity
    closest_maturity = min(tickers.keys(), key=lambda x: abs(x - maturity_in_years))
    ticker = tickers[closest_maturity]

    # Fetch and return the yield
    risk_free_rate = fetch_yield(ticker)

    if risk_free_rate is not None:
        print(f"Selected ticker: {ticker} for maturity {maturity_in_years:.2f} years")
        print(f"Risk-Free Rate: {risk_free_rate:.4%}")
    else:
        print(f"Failed to fetch yield for ticker: {ticker}")

    return risk_free_rate
