from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta


class FootprintScalping(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1m"

    # Strategy parameters
    absorption_volume_threshold = IntParameter(
        100, 1000, default=500, space="buy", optimize=True
    )
    absorption_price_threshold = DecimalParameter(
        0.01, 0.1, default=0.05, space="buy", optimize=True
    )
    bid_ask_ratio_threshold = DecimalParameter(
        1.5, 3.0, default=2.0, space="buy", optimize=True
    )
    value_area_percent = DecimalParameter(
        60, 80, default=70, space="buy", optimize=True
    )
    ema_period = IntParameter(21, 42, default=21, space="buy", optimize=True)

    poc_histogram_bins = IntParameter(50, 200, default=50, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate 21 EMA
        dataframe["ema"] = ta.EMA(dataframe, timeperiod=self.ema_period.value)

        # Add date-related columns
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe["day"] = dataframe["date"].dt.date

        # Group by day and calculate daily indicators
        dataframe = (
            dataframe.groupby("day")
            .apply(self.calculate_daily_indicators)
            .reset_index(level=0, drop=True)
        )

        # Detect exhaustive absorption
        dataframe["exhaustive_absorption"] = self.detect_exhaustive_absorption(
            dataframe
        )

        # Calculate price and open interest changes
        dataframe["price_change"] = dataframe["close"].pct_change()
        dataframe["oi_change"] = dataframe["open_interest"].pct_change()

        return dataframe

    def calculate_daily_indicators(self, group):
        # Calculate VWAP
        group["vwap"] = (
            group["volume"] * (group["high"] + group["low"] + group["close"]) / 3
        ).cumsum() / group["volume"].cumsum()

        # Calculate POC, VAH, and VAL using histogram approach
        close = group["close"]
        volume = group["volume"]

        # Create volume-price histogram
        hist, bin_edges = np.histogram(
            close, bins=self.poc_histogram_bins.value, weights=volume
        )

        # Calculate POC
        poc_index = np.argmax(hist)

        group["poc_lower"] = bin_edges[poc_index]
        group["poc_upper"] = bin_edges[poc_index + 1]

        # Calculate VAH and VAL
        total_volume = np.sum(hist)
        cumulative_volume = np.cumsum(hist)
        value_area_threshold = total_volume * self.value_area_percent.value / 100

        vah_index = np.argmax(cumulative_volume > (total_volume - value_area_threshold))
        val_index = np.argmax(cumulative_volume >= value_area_threshold)

        group["vah"] = bin_edges[vah_index]

        group["val"] = bin_edges[val_index]

        # Calculate bull and bear POC
        delta = group["volume"] * (2 * (group["close"] > group["open"]) - 1)
        bull_delta = np.where(delta > 0, delta, 0)
        bear_delta = np.where(delta < 0, -delta, 0)

        bull_hist, _ = np.histogram(
            close, bins=self.poc_histogram_bins.value, weights=bull_delta
        )
        bear_hist, _ = np.histogram(
            close, bins=self.poc_histogram_bins.value, weights=bear_delta
        )

        bull_poc_index = np.argmax(bull_hist)
        bear_poc_index = np.argmax(bear_hist)

        group["bull_poc_lower"] = bin_edges[bull_poc_index]
        group["bull_poc_upper"] = bin_edges[bull_poc_index + 1]

        group["bear_poc_lower"] = bin_edges[bear_poc_index]
        group["bear_poc_upper"] = bin_edges[bear_poc_index + 1]

        return group

    def detect_exhaustive_absorption(self, dataframe: DataFrame) -> pd.Series:
        # Calculate the ratio of bid volume to ask volume
        volume_ratio = dataframe["bid_volume"] / dataframe["ask_volume"]

        # Detect high volume
        high_volume = dataframe["volume"] > self.absorption_volume_threshold.value

        # Detect small price movement
        small_price_move = (
            dataframe["price_change"].abs() < self.absorption_price_threshold.value
        )

        # Determine trend based on EMA and daily VWAP
        uptrend = dataframe["ema"] > dataframe["vwap"]
        downtrend = dataframe["ema"] < dataframe["vwap"]

        # Detect potential downtrend reversal (high selling pressure absorbed in a downtrend)
        downtrend_reversal = downtrend & (
            volume_ratio > self.bid_ask_ratio_threshold.value
        )

        # Detect potential uptrend reversal (high buying pressure absorbed in an uptrend)
        uptrend_reversal = uptrend & (
            volume_ratio < 1 / self.bid_ask_ratio_threshold.value
        )

        # Combine conditions
        exhaustive_absorption = (
            high_volume & small_price_move & (downtrend_reversal | uptrend_reversal)
        ).astype(int)

        # Add labels for the type of absorption
        exhaustive_absorption = exhaustive_absorption.where(
            ~downtrend_reversal, other=-exhaustive_absorption
        )  # Negative for downtrend reversal

        return exhaustive_absorption

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Define price ranges
        in_val = dataframe["low"] <= dataframe["val"]
        in_vah = dataframe["high"] >= dataframe["vah"]

        in_poc_bin = (dataframe["close"] >= dataframe["poc_lower"]) & (
            dataframe["close"] < dataframe["poc_upper"]
        )
        in_bull_poc_bin = (dataframe["close"] >= dataframe["bull_poc_lower"]) & (
            dataframe["close"] < dataframe["bull_poc_upper"]
        )
        in_bear_poc_bin = (dataframe["close"] >= dataframe["bear_poc_lower"]) & (
            dataframe["close"] < dataframe["bear_poc_upper"]
        )

        # Enter long on downtrend reversal
        long_condition = (dataframe["exhaustive_absorption"] < 0) & (
            in_val
            | in_poc_bin
            | in_bull_poc_bin
            | (dataframe["close"] >= dataframe["val"])
        )
        dataframe.loc[long_condition, "enter_long"] = 1
        dataframe.loc[long_condition, "enter_tag"] = (
            dataframe.loc[long_condition, "enter_tag"] + "Exhaustive_Absorption_Long "
        )

        # Enter short on uptrend reversal
        short_condition = (dataframe["exhaustive_absorption"] > 0) & (
            in_vah
            | in_poc_bin
            | in_bear_poc_bin
            | (dataframe["close"] <= dataframe["vah"])
        )
        dataframe.loc[short_condition, "enter_short"] = 1
        dataframe.loc[short_condition, "enter_tag"] = (
            dataframe.loc[short_condition, "enter_tag"] + "Exhaustive_Absorption_Short "
        )

        # Add specific tags for each price level
        dataframe.loc[long_condition & in_val, "enter_tag"] += "VAL "
        dataframe.loc[long_condition & in_poc_bin, "enter_tag"] += "POC "
        dataframe.loc[long_condition & in_bull_poc_bin, "enter_tag"] += "BullPOC "
        dataframe.loc[short_condition & in_vah, "enter_tag"] += "VAH "
        dataframe.loc[short_condition & in_poc_bin, "enter_tag"] += "POC "
        dataframe.loc[short_condition & in_bear_poc_bin, "enter_tag"] += "BearPOC "

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Exit long positions
        long_exit_condition = (
            (dataframe["price_change"] > 0) & (dataframe["oi_change"] < 0)
        ) | (  # shorts closing
            (dataframe["price_change"] < 0) & (dataframe["oi_change"] < 0)
        )  # longs closing
        dataframe.loc[long_exit_condition, "exit_long"] = 1

        # Exit short positions
        short_exit_condition = (
            (dataframe["price_change"] > 0) & (dataframe["oi_change"] > 0)
        ) | (  # longs opening
            (dataframe["price_change"] < 0) & (dataframe["oi_change"] > 0)
        )  # shorts opening
        dataframe.loc[short_exit_condition, "exit_short"] = 1

        # Add exit tags for analysis
        dataframe.loc[
            long_exit_condition & (dataframe["price_change"] > 0), "exit_tag"
        ] += "ShortsClosed "
        dataframe.loc[
            long_exit_condition & (dataframe["price_change"] < 0), "exit_tag"
        ] += "LongsClosed "
        dataframe.loc[
            short_exit_condition & (dataframe["price_change"] > 0), "exit_tag"
        ] += "LongsOpened "
        dataframe.loc[
            short_exit_condition & (dataframe["price_change"] < 0), "exit_tag"
        ] += "ShortsOpened "

        return dataframe
