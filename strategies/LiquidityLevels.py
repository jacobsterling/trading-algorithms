from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from typing import Optional


class LiquidityLevels(IStrategy):
    """
    LiquidityLevels Strategy

    This strategy is based on the Liquidity Swings concept, identifying potential
    entry and exit points using pivot highs and lows, along with volume analysis.

    Key features:
    1. Identifies pivot highs and lows
    2. Calculates liquidity areas based on pivot points
    3. Uses volume analysis to filter significant levels
    4. Implements entry and exit rules based on price interaction with liquidity levels
    """

    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True

    stoploss = -0.10

    # Strategy parameters
    pivot_length = IntParameter(10, 50, default=14, space="buy", optimize=True)
    volume_filter = DecimalParameter(0, 2, default=0.5, space="buy", optimize=True)
    atr_period = IntParameter(5, 25, default=14, space="sell", optimize=True)

    # Risk parameters
    risk_percent = 0.05  # Risk 5% of the trading balance per trade
    max_trades = IntParameter(1, 20, default=10, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate pivot highs and lows
        dataframe["pivot_high"] = self.calculate_pivots(
            dataframe, self.pivot_length.value, "high"
        )
        dataframe["pivot_low"] = self.calculate_pivots(
            dataframe, self.pivot_length.value, "low"
        )

        # Calculate volume profile
        dataframe["volume_ma"] = (
            dataframe["volume"].rolling(window=self.pivot_length.value).mean()
        )

        # Identify liquidity areas
        dataframe["high_liquidity"] = (dataframe["high"] == dataframe["pivot_high"]) & (
            dataframe["volume"] > dataframe["volume_ma"] * self.volume_filter.value
        )
        dataframe["low_liquidity"] = (dataframe["low"] == dataframe["pivot_low"]) & (
            dataframe["volume"] > dataframe["volume_ma"] * self.volume_filter.value
        )

        # Calculate ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        return dataframe

    def calculate_pivots(
        self, dataframe: DataFrame, length: int, pivot_type: str
    ) -> pd.Series:
        if pivot_type not in ["high", "low"]:
            raise ValueError("pivot_type must be either 'high' or 'low'")

        price_col = pivot_type
        pivots = dataframe[price_col].copy()

        for i in range(len(dataframe)):
            start = max(0, i - length)
            end = i + 1
            if pivot_type == "high":
                pivots.iloc[i] = dataframe[price_col].iloc[start:end].max()
            else:
                pivots.iloc[i] = dataframe[price_col].iloc[start:end].min()

        # Reset pivot when price breaks through
        if pivot_type == "high":
            reset_mask = dataframe[price_col] > pivots
        else:
            reset_mask = dataframe[price_col] < pivots

        pivots[reset_mask] = dataframe.loc[reset_mask, price_col]

        return pivots

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["low"] < dataframe["pivot_low"]) & (dataframe["low_liquidity"]),
            ["enter_long", "enter_tag"],
        ] = (1, "long_liquidity_bounce")

        dataframe.loc[
            (dataframe["high"] > dataframe["pivot_high"])
            & (dataframe["high_liquidity"]),
            ["enter_short", "enter_tag"],
        ] = (1, "short_liquidity_bounce")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["high"] > dataframe["pivot_high"])
            | (dataframe["high_liquidity"]),
            ["exit_long", "exit_tag"],
        ] = (1, "long_exit_liquidity")

        dataframe.loc[
            (dataframe["low"] < dataframe["pivot_low"]) | (dataframe["low_liquidity"]),
            ["exit_short", "exit_tag"],
        ] = (1, "short_exit_liquidity")

        return dataframe

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
        Custom stoploss logic, returning the new stoploss as a percentage.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Determine the liquidity level based on trade direction
        if trade.is_short:
            liquidity_level = last_candle["pivot_high"]
        else:
            liquidity_level = last_candle["pivot_low"]

        # Calculate the stoploss price
        atr = last_candle["atr"]
        if trade.is_short:
            stoploss_price = liquidity_level + atr
        else:
            stoploss_price = liquidity_level - atr

        # Calculate the stoploss percentage
        stoploss_percent = (stoploss_price / current_rate - 1) * 100

        # Ensure the stoploss is positive and not too large
        stoploss_percent = max(min(stoploss_percent, 25), 0.1)

        return stoploss_percent

    def custom_entry_price(
        self,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """
        Custom entry price logic, returning the new entry price.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if entry_tag == "long_liquidity_bounce":
            return last_candle["pivot_low"]
        elif entry_tag == "short_liquidity_bounce":
            return last_candle["pivot_high"]
        else:
            return proposed_rate

    def _calculate_position_size(
        self, pair: str, current_time: datetime, current_rate: float, side: str
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Calculate the stoploss percentage
        if self.dp.runmode.value in ("live", "dry_run"):
            stoploss_percent = abs(
                self.custom_stoploss(pair, None, current_time, current_rate, 0.0)
            )
        else:
            # For backtesting, use the ATR-based stoploss
            stoploss_price = (
                last_candle["close"] - last_candle["atr"]
                if side == "long"
                else last_candle["close"] + last_candle["atr"]
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
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
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

    @property
    def max_entry_position_adjustment(self):
        """
        Limits the number of times we can adjust the entry position.
        """
        return -1  # Disable position adjustment
