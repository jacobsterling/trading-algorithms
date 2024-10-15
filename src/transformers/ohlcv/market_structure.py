import numpy as np
from pandas import DataFrame

PIVOT = "pivot"
HH = "HH"
LH = "LH"
HL = "HL"
LL = "LL"

BOS = "BOS"


def detect_pivot_points(data: DataFrame, window: int) -> np.ndarray:
    """
    Detect pivot points in the market structure and classify them as HH, LH, HL, or LL.

    :param data: DataFrame containing 'high' and 'low' columns
    :param window: The number of periods to look before and after for pivot detection
    :return: DataFrame with a 'pivot' column containing pivot classifications
    """
    pivot = np.full(len(data), np.nan, dtype=object)
    last_high, last_low = None, None
    first_high, first_low = True, True

    for i in range(window, len(data) - window):
        if all(data["high"].iloc[i] > data["high"].iloc[i - window : i]) and all(
            data["high"].iloc[i] > data["high"].iloc[i + 1 : i + window + 1]
        ):
            if first_high:
                first_high = False
            elif last_high is None or data["high"].iloc[i] > last_high:
                pivot[i] = HH
            else:
                pivot[i] = LH
            last_high = data["high"].iloc[i]

        if all(data["low"].iloc[i] < data["low"].iloc[i - window : i]) and all(
            data["low"].iloc[i] < data["low"].iloc[i + 1 : i + window + 1]
        ):
            if first_low:
                first_low = False
            elif last_low is None or data["low"].iloc[i] > last_low:
                pivot[i] = HL
            else:
                pivot[i] = LL
            last_low = data["low"].iloc[i]

    data[PIVOT] = pivot

    return data


def detect_market_structure_breaks(data, window=3, column=PIVOT):
    """
    Detect breaks in market structure based on pivot points.

    :param data: DataFrame containing 'pivot' column
    :param window: Number of pivots to look back to determine market structure
    :return: DataFrame with a new 'BOS' (Break of Structure) column
    """
    pivot_indices = data.index[data[column].notna()]

    for i in range(window, len(pivot_indices)):
        current_index = pivot_indices[i]
        previous_pivots = data.loc[pivot_indices[i - window : i], column]

        bearish_count = sum(previous_pivots.isin([LH, LL]))
        bullish_count = sum(previous_pivots.isin([HH, HL]))

        current_pivot = data.loc[current_index, column]

        if bearish_count > bullish_count and current_pivot in [HH, HL]:
            data.loc[current_index, BOS] = "bullish_BOS"
        elif bullish_count > bearish_count and current_pivot in [LH, LL]:
            data.loc[current_index, BOS] = "bearish_BOS"

    return data


def find_liquidity_levels(
    data: DataFrame, tolerance_percent: float = 0.01
) -> DataFrame:
    """
    Find liquidity levels in the market structure based on pivot points.

    :param data: DataFrame containing 'pivot' column and OHLC data
    :param tolerance_percent: Percentage tolerance for grouping similar price levels
    :return: DataFrame with liquidity levels information
    """
    levels = []
    pivot_data = data[data[PIVOT].notna()]

    def process_pivot(pivot, column):
        current_level = pivot[column]
        tolerance = current_level * (tolerance_percent / 100)
        similar_points = pivot_data[
            (
                pivot_data[PIVOT].isin([LL, HL])
                if column == "low"
                else pivot_data[PIVOT].isin([HH, LH])
            )
            & (
                pivot_data[column].between(
                    current_level - tolerance, current_level + tolerance
                )
            )
        ]

        if len(similar_points) >= 2:
            end_index = similar_points.index[-1]
            for j in range(data.index.get_loc(similar_points.index[0]) + 1, len(data)):
                condition = (
                    data[column].iloc[j] < current_level - tolerance
                    if column == "low"
                    else data[column].iloc[j] > current_level + tolerance
                )
                if condition:
                    end_index = data.index[j - 1]
                    break

            return {
                "price": current_level,
                "start": similar_points.index[0],
                "end": end_index,
                "type": column,
            }

    for _, pivot in pivot_data.iterrows():
        if pivot[PIVOT] in [LL, HL]:
            level = process_pivot(pivot, "low")
        elif pivot[PIVOT] in [HH, LH]:
            level = process_pivot(pivot, "high")
        else:
            continue

        if level:
            levels.append(level)

    # Remove duplicates
    unique_levels = []
    for level in levels:
        if not any(
            np.isclose(level["price"], l["price"], rtol=tolerance_percent / 100)
            and level["type"] == l["type"]
            for l in unique_levels
        ):
            unique_levels.append(level)

    return DataFrame(unique_levels)


def detect_cvd_pivot_points(
    data: DataFrame, window: int, column: str = "cvd"
) -> DataFrame:
    """
    Detect pivot points in the CVD (or any single-value series) and classify them as HH, LH, HL, or LL.

    :param data: DataFrame containing the column to analyze (e.g., 'cvd')
    :param window: The number of periods to look before and after for pivot detection
    :param column: The name of the column to analyze (default is 'cvd')
    :return: DataFrame with a 'pivot' column containing pivot classifications
    """
    pivot = np.full(len(data), np.nan, dtype=object)
    last_high, last_low = None, None
    first_high, first_low = True, True

    for i in range(window, len(data) - window):
        current_value = data[column].iloc[i]

        # Check for local maximum
        if all(current_value > data[column].iloc[i - window : i]) and all(
            current_value > data[column].iloc[i + 1 : i + window + 1]
        ):
            if first_high:
                first_high = False
            elif last_high is None or current_value > last_high:
                pivot[i] = HH
            else:
                pivot[i] = LH
            last_high = current_value

        # Check for local minimum
        if all(current_value < data[column].iloc[i - window : i]) and all(
            current_value < data[column].iloc[i + 1 : i + window + 1]
        ):
            if first_low:
                first_low = False
            elif last_low is None or current_value < last_low:
                pivot[i] = LL
            else:
                pivot[i] = HL
            last_low = current_value

    data[f"{column}_{PIVOT}"] = pivot

    return data
