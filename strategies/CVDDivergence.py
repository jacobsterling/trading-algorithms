from freqtrade.strategy import (
    IStrategy,
    CategoricalParameter,
    IntParameter,
    DecimalParameter,
)
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import talib.abstract as ta
from datetime import datetime
from freqtrade.persistence import Trade


class CVDDivergence(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1m"
    trailing_stop = False
    process_only_new_candles = False
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count: int = 288
    can_short = True

    # Add hyperparameters
    pivot_window = IntParameter(5, 15, default=5, space="buy", optimize=True)
    divergence_threshold = DecimalParameter(
        0.01, 2.0, default=1.266, space="buy", optimize=True
    )
    cvd_ma_window = IntParameter(20, 40, default=22, space="buy", optimize=True)

    # Add ATR period and multiplier parameters
    atr_period = IntParameter(5, 25, default=14, space="sell", optimize=True)
    atr_multiplier = DecimalParameter(
        1.0, 4.0, default=2.0, space="sell", optimize=True
    )

    # Change risk_amount to a percentage
    risk_percent = 0.05  # Risk 5% of the trading balance per trade

    max_trades = IntParameter(1, 9, default=1, space="buy", optimize=True)

    poc_histogram_bins = IntParameter(10, 100, default=50, space="buy", optimize=True)

    @property
    def _allowance_per_trade(self) -> float:

        if self.wallets is None:
            trading_balance = 1000  # Default dry_run_wallet amount
        else:
            trading_balance = self.wallets.get_total_stake_amount()

        allowance_per_trade = trading_balance / self.max_trades.value

        return allowance_per_trade

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe["day"] = dataframe["date"].dt.date

        dataframe.set_index("date", inplace=True)

        dataframe = (
            dataframe.groupby("day")
            .apply(self._daily_calculations)
            .reset_index(level=0, drop=True)
        )

        dataframe.reset_index(inplace=True)

        # Calculate ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = (
            dataframe.groupby("day")
            .apply(self._process_day_entry)
            .reset_index(level=0, drop=True)
        )

        # Ensure consistent data types for enter_long and enter_short columns
        dataframe["enter_long"] = dataframe["enter_long"].astype(float)
        dataframe["enter_short"] = dataframe["enter_short"].astype(float)

        # Calculate dynamic stoploss based on ATR
        dataframe["atr_stoploss_long"] = dataframe["close"] - (
            dataframe["atr"] * self.atr_multiplier.value
        )
        dataframe["atr_stoploss_short"] = dataframe["close"] + (
            dataframe["atr"] * self.atr_multiplier.value
        )

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # dataframe.loc[dataframe["divergence"] < 0, ["exit_long", "exit_tag"]] = (
        #     1,
        #     "Bear_Div",
        # )

        dataframe.loc[
            dataframe["high"] >= dataframe["vwap_upper_2"],
            ["exit_long", "exit_short", "exit_tag"],
        ] = (1, 1, "VWAP_Upper")

        long_poc_exit = dataframe["bear_poc_lower"] >= dataframe["vwap_upper_1"]

        dataframe.loc[
            (dataframe["high"] >= dataframe["bear_poc_lower"]) & long_poc_exit,
            ["exit_long", "exit_tag"],
        ] = (1, "POC_Upper")

        # dataframe.loc[dataframe["divergence"] > 0, ["exit_short", "exit_tag"]] = (
        #     1,
        #     "Bull_Div",
        # )

        dataframe.loc[
            dataframe["low"] <= dataframe["vwap_lower_2"],
            ["exit_short", "exit_long", "exit_tag"],
        ] = (1, 1, "VWAP_Lower")

        short_poc_exit = dataframe["bull_poc_upper"] <= dataframe["vwap_lower_1"]

        dataframe.loc[
            (dataframe["low"] <= dataframe["bull_poc_upper"]) & short_poc_exit,
            ["exit_short", "exit_tag"],
        ] = (1, "POC_Lower")

        return dataframe

    def _process_day_entry(self, group: DataFrame) -> DataFrame:
        # Ensure the function returns a DataFrame with consistent data types
        group = group.copy()

        # Initialize enter_long and enter_short columns
        group["enter_long"] = np.nan
        group["enter_short"] = np.nan

        # Find the last bullish and bearish divergences
        last_bull_div = group[group["divergence"] > 0].index.max()
        last_bear_div = group[group["divergence"] < 0].index.max()

        # Process entries after the last divergence
        if last_bull_div is not pd.NaT:
            bull_slice = group.loc[last_bull_div:]
            bull_invalidated = (
                (bull_slice["low"] <= bull_slice["vwap_lower_2"]).cummax().astype(bool)
            )
            valid_poc = bull_slice["bull_poc_upper"] <= bull_slice["vwap"]
            long_entry = (
                (bull_slice["low"] <= bull_slice["bull_poc_upper"])
                & ~bull_invalidated
                & valid_poc
            )
            group.loc[bull_slice.index[long_entry], ["enter_long", "enter_tag"]] = (
                1,
                "Bull_POC",
            )

        if last_bear_div is not pd.NaT:
            bear_slice = group.loc[last_bear_div:]
            bear_invalidated = (
                (bear_slice["high"] >= bear_slice["vwap_upper_2"]).cummax().astype(bool)
            )
            valid_poc = bear_slice["bear_poc_lower"] <= bear_slice["vwap"]
            short_entry = (
                (bear_slice["high"] >= bear_slice["bear_poc_lower"])
                & ~bear_invalidated
                & valid_poc
            )
            group.loc[bear_slice.index[short_entry], ["enter_short", "enter_tag"]] = (
                1,
                "Bear_POC",
            )

        return group

    @staticmethod
    def _find_pivot(series: Series, left: int, right: int, compare_func) -> Series:
        return series.rolling(window=left + right + 1, center=True).apply(
            lambda x: compare_func(x) == left
        )

    @staticmethod
    def _detect_divergence(
        dataframe: DataFrame, pivot_col: str, price_col: str, cvd_condition: str
    ) -> Series:
        return (dataframe[pivot_col] == 1) & (
            dataframe[price_col].diff().fillna(0) * cvd_condition
        )

    def _calculate_poc(self, group: DataFrame) -> tuple:

        close = group["close"]
        delta = group["delta"]

        # Separate bullish and bearish deltas
        bull_delta = np.where(
            delta > 0, delta, 0
        )  # Positive delta indicates buying pressure
        bear_delta = -np.where(
            delta < 0, delta, 0
        )  # Negative delta indicates selling pressure

        # Create price-delta histograms for bull and bear activity
        bull_hist, bull_bin_edges = np.histogram(
            close, bins=self.poc_histogram_bins.value, weights=bull_delta
        )
        bear_hist, bear_bin_edges = np.histogram(
            close, bins=self.poc_histogram_bins.value, weights=bear_delta
        )

        # Find the price bins with the highest total delta for bull and bear activity
        bull_max_bin_index = np.argmax(bull_hist)
        bear_max_bin_index = np.argmax(bear_hist)

        # Calculate the bin bounds for bull and bear POCs
        bull_poc_lower = bull_bin_edges[bull_max_bin_index]
        bull_poc_upper = bull_bin_edges[bull_max_bin_index + 1]
        bear_poc_lower = bear_bin_edges[bear_max_bin_index]
        bear_poc_upper = bear_bin_edges[bear_max_bin_index + 1]

        return (
            bear_poc_lower,
            bear_poc_upper,
            bull_poc_lower,
            bull_poc_upper,
        )

    def _detect_regular_bullish_divergence(self, dataframe: DataFrame) -> Series:
        return self._detect_divergence(dataframe, "pivot_low", "low", 1) & (
            dataframe["cvd"].diff().fillna(0) > self.divergence_threshold.value
        )

    def _detect_hidden_bullish_divergence(self, dataframe: DataFrame) -> Series:
        return self._detect_divergence(dataframe, "pivot_low", "low", -1) & (
            dataframe["cvd"].diff().fillna(0) < -self.divergence_threshold.value
        )

    def _detect_regular_bearish_divergence(self, dataframe: DataFrame) -> Series:
        return self._detect_divergence(dataframe, "pivot_high", "high", 1) & (
            dataframe["cvd"].diff().fillna(0) < -self.divergence_threshold.value
        )

    def _detect_hidden_bearish_divergence(self, dataframe: DataFrame) -> Series:
        return self._detect_divergence(dataframe, "pivot_high", "high", -1) & (
            dataframe["cvd"].diff().fillna(0) > self.divergence_threshold.value
        )

    def _calculate_vwap_and_bands(self, group: DataFrame) -> DataFrame:
        typical_price = (group["high"] + group["low"] + group["close"]) / 3

        cumulative_tp_v = (typical_price * group["volume"]).cumsum()
        cumulative_volume = group["volume"].cumsum()
        group["vwap"] = cumulative_tp_v / cumulative_volume
        squared_diff = (typical_price - group["vwap"]) ** 2
        std = np.sqrt((squared_diff * group["volume"]).cumsum() / cumulative_volume)

        group["vwap_upper_1"] = group["vwap"] + std
        group["vwap_lower_1"] = group["vwap"] - std
        group["vwap_upper_2"] = group["vwap"] + 2 * std
        group["vwap_lower_2"] = group["vwap"] - 2 * std

        return group

    def _daily_calculations(self, group: DataFrame) -> DataFrame:

        # Calculate CVD and related indicators
        group["cvd"] = group["delta"].cumsum()

        # Calculate pivot points with optimizable window
        group["pivot_low"] = self._find_pivot(
            group["cvd"],
            self.pivot_window.value // 2,
            self.pivot_window.value // 2,
            np.argmin,
        )
        group["pivot_high"] = self._find_pivot(
            group["cvd"],
            self.pivot_window.value // 2,
            self.pivot_window.value // 2,
            np.argmax,
        )

        # Calculate CVD moving average with optimizable window
        group["cvd_ma"] = (
            group["cvd"].rolling(window=self.cvd_ma_window.value, min_periods=1).mean()
        )

        group["divergence"] = np.select(
            [
                self._detect_regular_bullish_divergence(group),
                self._detect_hidden_bullish_divergence(group),
                self._detect_regular_bearish_divergence(group),
                self._detect_hidden_bearish_divergence(group),
            ],
            [2, 1, -2, -1],
            default=0,
        )

        # Calculate VWAP and bands
        group = self._calculate_vwap_and_bands(group)

        (
            group["bull_poc_lower"],
            group["bull_poc_upper"],
            group["bear_poc_lower"],
            group["bear_poc_upper"],
        ) = self._calculate_poc(group)

        return group

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if trade.is_short:
            stoploss_price = last_candle["atr_stoploss_short"]
            stoploss = (stoploss_price / current_rate) - 1
        else:
            stoploss_price = last_candle["atr_stoploss_long"]
            stoploss = 1 - (stoploss_price / current_rate)

        return stoploss

    def _calculate_position_size(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        side: str,
    ) -> float:  # Changed return type to float
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
                last_candle["atr_stoploss_long"]
                if side == "long"
                else last_candle["atr_stoploss_short"]
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
        **kwargs,
    ) -> float:
        side = "long" if proposed_stake > 0 else "short"

        position_size = self._calculate_position_size(
            pair, current_time, current_rate, side
        )

        # Ensure the position size is within allowed limits
        fixed_position_size = min(max(position_size, min_stake), max_stake)

        return fixed_position_size

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
