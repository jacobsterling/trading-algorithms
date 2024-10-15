from pandas import DataFrame, Series
import numpy as np


def detect_divergence(self, dataframe: DataFrame) -> Series:

    def _calc_divergence(
        dataframe: DataFrame, pivot_col: str, price_col: str, cvd_condition: str
    ) -> Series:
        """
        Detects divergences between price and CVD.
        Used for both regular and hidden divergences, bullish and bearish.
        """
        return (dataframe[pivot_col] == 1) & (
            dataframe[price_col].diff().fillna(0) * cvd_condition
        )

    _PIVOT_COL = "pivot_low"
    _PRICE_COL = "low"

    _DIV_THRESHOLD = self.divergence_threshold.value

    def _detect_regular_bullish_divergence(dataframe: DataFrame) -> Series:
        return _calc_divergence(dataframe, _PIVOT_COL, _PRICE_COL, 1) & (
            dataframe["cvd"].diff().fillna(0) > _DIV_THRESHOLD
        )

    def _detect_hidden_bullish_divergence(dataframe: DataFrame) -> Series:
        return _calc_divergence(dataframe, _PIVOT_COL, _PRICE_COL, -1) & (
            dataframe["cvd"].diff().fillna(0) < -_DIV_THRESHOLD
        )

    _PIVOT_COL = "pivot_high"
    _PRICE_COL = "high"

    def _detect_regular_bearish_divergence(self, dataframe: DataFrame) -> Series:
        return _calc_divergence(dataframe, _PIVOT_COL, _PRICE_COL, 1) & (
            dataframe["cvd"].diff().fillna(0) < -_DIV_THRESHOLD
        )

    def _detect_hidden_bearish_divergence(self, dataframe: DataFrame) -> Series:
        return _calc_divergence(dataframe, _PIVOT_COL, _PRICE_COL, -1) & (
            dataframe["cvd"].diff().fillna(0) > _DIV_THRESHOLD
        )

    # Calculate basic divergence
    dataframe["divergence"] = np.select(
        [
            _detect_regular_bullish_divergence(dataframe),
            _detect_hidden_bullish_divergence(dataframe),
            _detect_regular_bearish_divergence(dataframe),
            _detect_hidden_bearish_divergence(dataframe),
        ],
        [2, 1, -2, -1],
        default=0,
    )

    def _confirm_divergence(divergence: Series) -> Series:
        return (
            divergence.rolling(window=self.divergence_confirmation_window.value)
            .sum()
            .abs()
            >= self.divergence_confirmation_window.value
        )

    # Confirm divergence over multiple periods
    confirm_divergence = _confirm_divergence(dataframe["divergence"])

    def _confirm_with_rsi(dataframe: DataFrame) -> Series:
        bullish_confirmation = (dataframe["divergence"] > 0) & (
            dataframe["rsi"] < self.rsi_oversold.value
        )
        bearish_confirmation = (dataframe["divergence"] < 0) & (
            dataframe["rsi"] > self.rsi_overbought.value
        )
        return bullish_confirmation | bearish_confirmation

    # Confirm with RSI
    rsi_confirmed = _confirm_with_rsi(dataframe)

    # Combine all signals
    confirmed_divergence = confirm_divergence & rsi_confirmed

    return confirmed_divergence
