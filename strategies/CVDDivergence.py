from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
)
from pandas import DataFrame, Series, notna
import pandas as pd
import numpy as np
import talib.abstract as ta
from datetime import datetime
from freqtrade.persistence import Trade
from typing import Optional, Tuple
from scipy import stats


class CVDDivergence(IStrategy):
    """
    CVDDivergence Strategy

    This strategy uses Cumulative Volume Delta (CVD) divergences to identify potential
    entry points in the market. It employs a multi-faceted approach:

    1. Entry:
       - Detects regular and hidden divergences between price and CVD.
       - Uses Point of Control (POC) from volume profile for entry timing.
       - Implements a scaled entry approach, splitting orders into multiple parts.

    2. Exit:
       - Employs a scaled exit strategy, taking profits at predetermined levels.
       - Uses Volume Weighted Average Price (VWAP) bands for full position exit.

    3. Risk Management:
       - Implements dynamic position sizing based on ATR for stop loss calculation.
       - Uses a fixed percentage of account balance for risk per trade.

    4. Additional Features:
       - Supports both long and short trades.
       - Uses hyperparameters for strategy optimization.
    """

    INTERFACE_VERSION = 3
    timeframe = "1m"
    trailing_stop = False
    process_only_new_candles = False
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count: int = 288
    can_short = True

    # Add hyperparameters

    # Add ATR period and multiplier parameters
    atr_period = IntParameter(5, 25, default=14, space="sell", optimize=True)
    atr_multiplier = DecimalParameter(
        1.0, 4.0, default=1.0, space="sell", optimize=True
    )

    # Change risk_amount to a percentage
    risk_percent = 0.05  # Risk 5% of the trading balance per trade

    max_trades = IntParameter(1, 20, default=10, space="buy", optimize=True)

    poc_histogram_bins = IntParameter(10, 300, default=50, space="buy", optimize=True)

    # Add a parameter for the number of scale-out steps
    scale_out_count = IntParameter(1, 5, default=3, space="sell", optimize=True)

    rsi_period = IntParameter(10, 20, default=14, space="buy", optimize=True)
    rsi_overbought = IntParameter(65, 85, default=70, space="buy", optimize=True)
    rsi_oversold = IntParameter(15, 35, default=30, space="buy", optimize=True)

    # Add new hyperparameters
    divergence_lookback = IntParameter(10, 100, default=50, space="buy", optimize=True)
    divergence_threshold = DecimalParameter(
        1.5, 3.0, default=2.0, space="buy", optimize=True
    )
    roc_period = IntParameter(1, 20, default=5, space="buy", optimize=True)

    def _calculate_vwap_and_bands(self, group: DataFrame) -> DataFrame:
        typical_price = (group["high"] + group["low"] + group["close"]) / 3
        cumulative_tp_v = (typical_price * group["volume"]).cumsum()
        cumulative_volume = group["volume"].cumsum()
        group["vwap"] = cumulative_tp_v / cumulative_volume
        squared_diff = (typical_price - group["vwap"]) ** 2
        std = np.sqrt((squared_diff * group["volume"]).cumsum() / cumulative_volume)

        for i in range(1, 3):
            group[f"vwap_upper_{i}"] = group["vwap"] + i * std
            group[f"vwap_lower_{i}"] = group["vwap"] - i * std

        return group

    def _daily_calculations(self, dataframe: DataFrame) -> DataFrame:
        dataframe.set_index("date", inplace=True)

        def _calculations(group: DataFrame) -> DataFrame:
            # Calculate CVD and related indicators
            group["cvd"] = group["delta"].cumsum()

            # Calculate VWAP and bands
            group = self._calculate_vwap_and_bands(group)

            (
                group["bull_poc_lower"],
                group["bull_poc_upper"],
                group["bear_poc_lower"],
                group["bear_poc_upper"],
            ) = self._calculate_poc_entry(group)

            return group

        return (
            dataframe.groupby("day")
            .apply(_calculations)
            .reset_index(level=0, drop=True)
            .reset_index()
        )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculates various indicators used in the strategy:
        - CVD (Cumulative Volume Delta)
        - Pivot points for CVD
        - VWAP and VWAP bands
        - ATR for dynamic stop loss
        - Point of Control (POC) from volume profile
        - Divergences
        """
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe["day"] = dataframe["date"].dt.date

        dataframe = self._daily_calculations(dataframe)

        # Calculate ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        # Add RSI calculation
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        return dataframe

    def detect_divergences(self, df: DataFrame) -> Tuple[Series, Series]:
        """
        Detect both bullish and bearish divergences using a quantitative approach.
        """
        lookback = self.divergence_lookback.value
        threshold = self.divergence_threshold.value
        roc_period = self.roc_period.value

        # Calculate rate of change
        df["price_roc"] = df["close"].pct_change(roc_period)
        df["cvd_roc"] = df["cvd"].pct_change(roc_period)

        bullish_div = np.zeros(len(df), dtype=bool)
        bearish_div = np.zeros(len(df), dtype=bool)

        for i in range(lookback, len(df)):
            price_window = df["close"].iloc[i - lookback : i]
            cvd_window = df["cvd"].iloc[i - lookback : i]

            # Find local min/max
            price_min_idx = price_window.idxmin()
            price_max_idx = price_window.idxmax()
            cvd_min_idx = cvd_window.idxmin()
            cvd_max_idx = cvd_window.idxmax()

            # Calculate slopes
            price_slope = (df["close"].iloc[i] - price_window.iloc[0]) / lookback
            cvd_slope = (df["cvd"].iloc[i] - cvd_window.iloc[0]) / lookback

            # Detect bullish divergence
            if (
                price_min_idx == price_window.index[-1]
                and cvd_min_idx != cvd_window.index[-1]
                and price_slope < 0
                and cvd_slope > 0
                and df["price_roc"].iloc[i] < -threshold
                and df["cvd_roc"].iloc[i] > threshold
            ):
                bullish_div[i] = True

            # Detect bearish divergence
            if (
                price_max_idx == price_window.index[-1]
                and cvd_max_idx != cvd_window.index[-1]
                and price_slope > 0
                and cvd_slope < 0
                and df["price_roc"].iloc[i] > threshold
                and df["cvd_roc"].iloc[i] < -threshold
            ):
                bearish_div[i] = True

        return Series(bullish_div, index=df.index), Series(bearish_div, index=df.index)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bullish_div, bearish_div = self.detect_divergences(dataframe)

        dataframe.loc[
            (bullish_div)
            & (dataframe["close"] < dataframe["vwap"])
            & (dataframe["low"] > dataframe["bear_poc_upper"])
            & (dataframe["rsi"] < self.rsi_oversold.value),
            ["enter_long", "enter_tag"],
        ] = (1, "Long_Divergence")

        dataframe.loc[
            (bearish_div)
            & (dataframe["close"] > dataframe["vwap"])
            & (dataframe["high"] < dataframe["bull_poc_lower"])
            & (dataframe["rsi"] > self.rsi_overbought.value),
            ["enter_short", "enter_tag"],
        ] = (1, "Short_Divergence")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generates exit signals for both scaled exits and full position exits.
        Uses POC levels for scaled exits and VWAP bands for full exits.
        """

        dataframe.loc[
            dataframe["high"] >= dataframe["vwap_upper_2"],
            [
                "exit_long",
                "exit_tag",
            ],
        ] = (1, "VWAP_Upper")

        dataframe.loc[
            dataframe["cvd"] >= 0,
            [
                "exit_long",
                "exit_tag",
            ],
        ] = (1, "Delta_Unwound")

        dataframe.loc[
            dataframe["low"] <= dataframe["vwap_lower_2"],
            [
                "exit_short",
                "exit_tag",
            ],
        ] = (1, "VWAP_Lower")

        dataframe.loc[
            dataframe["cvd"] <= 0,
            [
                "exit_short",
                "exit_tag",
            ],
        ] = (1, "Delta_Unwound")

        return dataframe

    def _calculate_poc_entry(self, group: DataFrame) -> tuple:
        high = group["high"]
        low = group["low"]
        close = group["close"]
        delta = group["delta"]

        # Create bins across the entire price range
        bins = np.linspace(low.min(), high.max(), self.poc_histogram_bins.value + 1)

        # Separate bullish and bearish deltas
        bull_delta = np.where(delta > 0, delta, 0)
        bear_delta = np.where(delta < 0, -delta, 0)

        # Create price-delta histograms for bull and bear activity
        bull_hist, _ = np.histogram(close, bins=bins, weights=bull_delta)
        bear_hist, _ = np.histogram(close, bins=bins, weights=bear_delta)

        # Find the price bins with the highest total delta for bull and bear activity
        bull_max_bin_index = np.argmax(bull_hist)
        bear_max_bin_index = np.argmax(bear_hist)

        # Calculate the bin bounds for bull and bear POCs
        bull_poc_lower = bins[bull_max_bin_index]
        bull_poc_upper = bins[bull_max_bin_index + 1]
        bear_poc_lower = bins[bear_max_bin_index]
        bear_poc_upper = bins[bear_max_bin_index + 1]

        return (
            bull_poc_lower,
            bull_poc_upper,
            bear_poc_lower,
            bear_poc_upper,
        )

    def custom_entry_price(
        self,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:

        last_candle = self.get_latest_candle(pair, self.timeframe)

        match entry_tag:
            case "Long_Divergence":
                # For long entries, use the bear POC upper level
                entry_price = last_candle["bear_poc_upper"]
            case "Short_Divergence":
                # For short entries, use the bull POC lower level
                entry_price = last_candle["bull_poc_lower"]
            case _:
                return proposed_rate

        # Ensure the entry price is valid (not NaN and positive)
        if pd.isna(entry_price) or entry_price <= 0:
            return proposed_rate

        return entry_price

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Implements a dynamic stop loss based on ATR.
        Adapts the stop loss level to market volatility.
        """
        last_candle = self.get_latest_candle(pair, self.timeframe)

        atr_multiplier = self.atr_multiplier.value

        # For short entries, use the bull POC lower level
        entry_price = (
            last_candle["bull_poc_lower"]
            if trade.is_short
            else last_candle["bear_poc_upper"]
        )

        # Calculate stoploss prices using the ATR
        if trade.is_short:
            stoploss_price = entry_price - (last_candle["atr"] * atr_multiplier)
            stoploss = (stoploss_price / current_rate) - 1
        else:
            stoploss_price = entry_price + (last_candle["atr"] * atr_multiplier)
            stoploss = 1 - (stoploss_price / current_rate)

        return stoploss

    def _calculate_position_size(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        side: str,
    ) -> float:
        last_candle = self.get_latest_candle(pair, self.timeframe)

        # Calculate the stoploss percentage
        if self.dp.runmode.value in ("live", "dry_run"):
            stoploss_percent = abs(
                self.custom_stoploss(pair, None, current_time, current_rate, 0.0)
            )
        else:

            atr_multiplier = self.atr_multiplier.value

            # For backtesting, use the ATR-based stoploss
            stoploss_price = (
                last_candle["close"] - (last_candle["atr"] * atr_multiplier)
                if side == "long"
                else last_candle["close"] + (last_candle["atr"] * atr_multiplier)
            )
            stoploss_percent = abs(1 - (stoploss_price / current_rate))

        # Ensure stoploss_percent is not zero to avoid division by zero
        stoploss_percent = max(stoploss_percent, 0.001)  # Minimum 0.1% stoploss

        # Calculate the risk amount based on the trading balance
        trading_balance = self.wallets.get_total_stake_amount()
        risk_amount = trading_balance * self.risk_percent

        # Calculate the position size based on the risk amount and stoploss
        position_size = risk_amount / stoploss_percent

        return position_size

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        entry_tag: Optional[str] = None,
        **kwargs,
    ) -> float:
        """
        Calculates the position size based on the risk percentage and ATR-based stop loss.
        Adjusts position size for scaled entries and exits.
        """
        return self._allowance_per_trade

    @property
    def _allowance_per_trade(self) -> float:

        if self.wallets is None:
            trading_balance = 1000  # Default dry_run_wallet amount
        else:
            trading_balance = self.wallets.get_total_stake_amount()

        allowance_per_trade = trading_balance / self.max_trades.value

        return allowance_per_trade

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:

        position_size = self._calculate_position_size(
            pair, current_time, current_rate, side
        )

        # Calculate required leverage
        required_leverage = position_size / self._allowance_per_trade

        # Ensure the leverage is within allowed limits
        leverage = min(max(required_leverage, 1), max_leverage)

        return leverage
